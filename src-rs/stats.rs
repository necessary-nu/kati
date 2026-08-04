/*
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//! `kati::stats` implements stats collection and reporting about regions of
//! execution.
//!
//! Collection sites used to expand to a `static` apiece, so the number of
//! process globals grew with the number of call sites. They now name an entry
//! in a session-owned [`StatsRegistry`], which the macros reach through
//! whatever [`Context`] is in scope.

use crate::session::{Context, Session};
use parking_lot::Mutex;
use std::{
    collections::{HashMap, HashSet},
    ffi::{OsStr, OsString},
    fmt::Display,
    sync::Arc,
    time::{Duration, Instant},
};

/// Every collection site a session has reached, in the order it first reached
/// them.
// [spec:ronin:req:make.no-ambient-state]
#[derive(Default)]
pub struct StatsRegistry {
    all: Mutex<Vec<Arc<Stats>>>,
}

impl StatsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// The entry named `name`, created on first use. There are a handful of
    /// distinct names, so a linear scan keeps the report in creation order
    /// without a second structure to hold it.
    fn get_or_create(&self, name: &'static str) -> Arc<Stats> {
        let mut all = self.all.lock();
        if let Some(found) = all.iter().find(|s| s.name == name) {
            return found.clone();
        }
        let stats = Arc::new(Stats::new(name));
        all.push(stats.clone());
        stats
    }

    fn take_all(&self) -> Vec<Arc<Stats>> {
        std::mem::take(&mut self.all.lock())
    }
}

#[derive(Default, Clone)]
struct StatsDetails {
    count: i64,
    elapsed: Duration,
}

/// `Stats` represents a single collection site.
pub struct Stats {
    name: &'static str,
    count: Mutex<i64>,
    elapsed: Mutex<Duration>,
    detailed: Mutex<HashMap<OsString, StatsDetails>>,
    interesting: Mutex<HashSet<OsString>>,
}

impl Stats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            elapsed: Mutex::new(Duration::new(0, 0)),
            count: Mutex::new(0),
            detailed: Mutex::new(HashMap::new()),
            interesting: Mutex::new(HashSet::new()),
        }
    }

    fn dump_top(&self) {
        let all_details = self.detailed.lock();
        if all_details.is_empty() {
            return;
        }

        let mut detailed = all_details
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>();
        detailed.sort_by_key(|b| std::cmp::Reverse(b.1.elapsed));
        // Only print the top 10
        detailed.truncate(10);

        let mut interesting = self.interesting.lock().clone();
        if !interesting.is_empty() {
            // No need to print anything out twice
            for (name, _) in detailed.iter() {
                interesting.remove(name);
            }

            for name in interesting {
                if let Some(details) = all_details.get(&name) {
                    detailed.push((name, details.clone()));
                } else {
                    detailed.push((name, StatsDetails::default()))
                }
            }
        }

        let max_cnt_len = detailed
            .iter()
            .map(|(_, v)| format!("{}", v.count).len())
            .max()
            .unwrap_or(1);
        for (name, details) in detailed {
            eprintln!(
                "*kati*: {:>6.3} / {:>max_cnt_len$} {}",
                details.elapsed.as_secs_f64(),
                details.count,
                name.to_string_lossy()
            );
        }
    }

    fn start(&self) -> Instant {
        let start = std::time::Instant::now();
        *self.count.lock() += 1;
        start
    }

    fn end(&self, start: Instant) -> Duration {
        let elapsed = start.elapsed();
        *self.elapsed.lock() += elapsed;
        elapsed
    }

    fn end_with_msg(&self, start: Instant, msg: &OsStr) -> Duration {
        let elapsed = start.elapsed();
        *self.elapsed.lock() += elapsed;
        let mut detailed = self.detailed.lock();
        if let Some(details) = detailed.get_mut(msg) {
            details.count += 1;
            details.elapsed += elapsed;
        } else {
            detailed.insert(msg.to_owned(), StatsDetails { count: 1, elapsed });
        }
        elapsed
    }

    /// Mark the specific execution as interesting. It will be logged even if
    /// it isn't in the top 10 executions.
    pub fn mark_interesting(&self, name: OsString) {
        self.interesting.lock().insert(name);
    }
}

impl Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let detailed = self.detailed.lock();
        if !detailed.is_empty() {
            return write!(
                f,
                "{}: {} / {} ({} unique)",
                self.name,
                self.elapsed.lock().as_secs_f64(),
                *self.count.lock(),
                detailed.len()
            );
        }
        write!(
            f,
            "{}: {} / {}",
            self.name,
            self.elapsed.lock().as_secs_f64(),
            *self.count.lock()
        )
    }
}

/// The implementation behind [`collect_stats!`]. Returns `None`, and records
/// nothing, unless `--kati_stats` is on.
#[doc(hidden)]
#[must_use]
pub fn start_scope(ctx: &impl Context, name: &'static str) -> Option<ScopedStatsRecorder> {
    if !ctx.flags().enable_stat_logs {
        return None;
    }
    Some(ScopedStatsRecorder::new(ctx.stats().get_or_create(name)))
}

/// The implementation behind [`collect_stats_with_slow_report!`].
#[doc(hidden)]
#[must_use]
pub fn start_scope_with_slow_report(
    ctx: &impl Context,
    name: &'static str,
    msg: &OsStr,
) -> Option<ScopedStatsRecorderWithSlowReport> {
    if !ctx.flags().enable_stat_logs {
        return None;
    }
    Some(ScopedStatsRecorderWithSlowReport::new(
        ctx.stats().get_or_create(name),
        msg.to_os_string(),
    ))
}

/// Mark the makefile `name` as interesting at the `included makefiles` site.
pub fn mark_interesting(ctx: &impl Context, site: &'static str, name: OsString) {
    ctx.stats().get_or_create(site).mark_interesting(name);
}

#[doc(hidden)]
pub struct ScopedStatsRecorder {
    st: Arc<Stats>,
    start: Instant,
}

impl ScopedStatsRecorder {
    fn new(st: Arc<Stats>) -> Self {
        let start = st.start();
        Self { st, start }
    }
}

impl Drop for ScopedStatsRecorder {
    fn drop(&mut self) {
        self.st.end(self.start);
    }
}

/// Define and collect statistics about this block of code.
///
/// We'll record both the count and duration of these blocks, and report them
/// when [`report_all_stats`] is called. The first argument is whatever carries
/// the session: an `Evaluator`, or the `Session` itself.
#[macro_export]
macro_rules! collect_stats {
    ($ctx:expr, $name:literal) => {
        let _ssr = $crate::stats::start_scope($ctx, $name);
    };
}

#[doc(hidden)]
pub struct ScopedStatsRecorderWithSlowReport {
    st: Arc<Stats>,
    msg: OsString,
    start: Instant,
}

impl ScopedStatsRecorderWithSlowReport {
    fn new(st: Arc<Stats>, msg: OsString) -> Self {
        let start = st.start();
        Self { st, msg, start }
    }
}

impl Drop for ScopedStatsRecorderWithSlowReport {
    fn drop(&mut self) {
        let dur = self.st.end_with_msg(self.start, &self.msg);
        if dur > Duration::from_secs(3) {
            eprintln!(
                "*kati*: slow {} ({}): {}",
                self.st.name,
                dur.as_secs_f64(),
                self.msg.to_string_lossy()
            )
        }
    }
}

/// Define and collect statistics about this block of code, with specific
/// executions identified by the `msg`.
///
/// We'll record both the count and duration of these blocks, and report them
/// when [`report_all_stats`] is called. In addition, the top ten specific
/// instances will have their duration and counts reported as well.
///
/// Any executions over 3 seconds will be logged as they happen.
#[macro_export]
macro_rules! collect_stats_with_slow_report {
    ($ctx:expr, $name:literal, $msg:expr) => {
        let _ssr = $crate::stats::start_scope_with_slow_report($ctx, $name, $msg);
    };
}

/// Report all the statistics to stderr, if `--enable_stat_log` is enabled.
pub fn report_all_stats(session: &Session) {
    let all_stats = session.stats.take_all();
    if session.flags.enable_stat_logs {
        for stats in all_stats {
            eprintln!("*kati*: {stats}");
            stats.dump_top();
        }
        eprintln!("*kati*: {} symbols", session.symtab.count());
        eprintln!(
            "*kati*: {} find nodes",
            session
                .find_node_count
                .load(std::sync::atomic::Ordering::Relaxed)
        );
    }
}
