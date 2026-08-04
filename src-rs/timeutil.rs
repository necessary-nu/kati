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

use std::time::Instant;

use crate::session::Context;

pub struct ScopedTimeReporter {
    name: String,
    start: Instant,
    enabled: bool,
}

impl ScopedTimeReporter {
    pub fn new(ctx: &impl Context, name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: std::time::Instant::now(),
            enabled: ctx.flags().enable_stat_logs,
        }
    }
}

impl Drop for ScopedTimeReporter {
    fn drop(&mut self) {
        if self.enabled {
            let dur = self.start.elapsed();
            eprintln!("*kati*: {}: {}", self.name, dur.as_secs_f64());
        }
    }
}
