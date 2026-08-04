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

//! Fails when a mutable process global reappears anywhere under `src-rs`.
//!
//! Removing the globals once was a cleanup; keeping them gone is a gate. This
//! is a source scan rather than a lint because Rust has none for this, and
//! because the shape that has to be caught — the statistics macros used to
//! expand to a `static` per call site — is invisible to anyone reading a macro
//! definition rather than its expansions. A text scan sees macro bodies.
//!
//! `const` items are never reported: they are compile-time values with no
//! identity. A `static` may be kept by writing `// no-globals-gate: <reason>`
//! on the line above it, which is how the two permitted ones are permitted.
//!
//! See `plan/decisions/session-owned-evaluation.md` and
//! `[spec:ronin:req:make.no-ambient-state]`.

use std::path::{Path, PathBuf};

/// A `static` item found in the source.
#[derive(Debug, PartialEq, Eq)]
struct StaticItem {
    file: String,
    line: usize,
    name: String,
    /// The reason on the `// no-globals-gate:` marker above it, if any.
    exemption: Option<String>,
}

/// The node that was supposed to have removed each historical global, so a
/// regression points at its own history.
const FORMER_OWNERS: &[(&str, &str)] = &[
    ("FLAGS", "kati-flags-value"),
    ("SYMTAB", "kati-session-value"),
    ("GLOBAL_VARS", "kati-session-value"),
    ("SHELL_SYM", "kati-wellknown-symbols"),
    ("ALLOW_RULES_SYM", "kati-wellknown-symbols"),
    ("KATI_READONLY_SYM", "kati-wellknown-symbols"),
    ("VARIABLES_SYM", "kati-wellknown-symbols"),
    ("KATI_SYMBOLS_SYM", "kati-wellknown-symbols"),
    ("MAKEFILE_LIST", "kati-wellknown-symbols"),
    ("DEFAULT_FILENAME", "kati-wellknown-symbols"),
    ("GLOB_CACHE", "kati-caches-session"),
    ("CACHE", "kati-caches-session"),
    ("FIND_EMULATOR", "kati-caches-session"),
    ("NODE_COUNT", "kati-caches-session"),
    ("COMMAND_RESULTS", "kati-observation-session"),
    ("USED_ENV_VARS", "kati-observation-session"),
    ("USED_UNDEFINED_VARS", "kati-observation-session"),
    ("SHELL_STATUS", "kati-observation-session"),
    ("ALL_STATS", "kati-observation-session"),
    ("STATS", "kati-observation-session"),
];

const MARKER: &str = "no-globals-gate:";

/// Every `static` item in `source`, with the marker above it if there is one.
///
/// A line scan, deliberately: it sees `static` items inside macro bodies, which
/// is where the statistics collection sites used to hide one per call site.
fn scan(file: &str, source: &str) -> Vec<StaticItem> {
    let mut found = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let Some(name) = static_item_name(line) else {
            continue;
        };
        // The marker sits somewhere in the contiguous block of comments and
        // attributes directly above the item, so that it can be one line of a
        // longer explanation and so that `#[global_allocator]` may come
        // between. A blank line ends the block.
        let mut exemption = None;
        for prev in lines[..i].iter().rev() {
            let prev = prev.trim();
            if prev.starts_with("//") {
                if let Some((_, reason)) = prev.split_once(MARKER) {
                    exemption = Some(reason.trim().to_string());
                    break;
                }
                continue;
            }
            if prev.starts_with('#') {
                continue;
            }
            break;
        }
        found.push(StaticItem {
            file: file.to_string(),
            line: i + 1,
            name,
            exemption,
        });
    }
    found
}

/// The name of the `static` item declared on `line`, if it declares one.
///
/// Only an item declaration counts: a `&'static` lifetime is not one, and
/// neither is the word appearing in prose or in the middle of an expression.
fn static_item_name(line: &str) -> Option<String> {
    let mut rest = line.trim_start();
    for vis in ["pub(crate) ", "pub(super) ", "pub(self) ", "pub "] {
        if let Some(stripped) = rest.strip_prefix(vis) {
            rest = stripped.trim_start();
            break;
        }
    }
    let rest = rest.strip_prefix("static")?;
    // `static` has to be a whole word: `staticky` is not a declaration.
    let mut rest = rest.strip_prefix(char::is_whitespace)?.trim_start();
    if let Some(stripped) = rest.strip_prefix("mut ") {
        rest = stripped.trim_start();
    }
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        return None;
    }
    // A declaration names a type.
    if !rest[name.len()..].trim_start().starts_with(':') {
        return None;
    }
    Some(name)
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("src-rs is readable") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src-rs")
}

fn all_statics() -> Vec<StaticItem> {
    let root = src_root();
    let mut files = Vec::new();
    rust_sources(&root, &mut files);
    files.sort();
    let mut found = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        let source = std::fs::read_to_string(&path).expect("source file is readable");
        found.extend(scan(&rel, &source));
    }
    found
}

/// No `static` under `src-rs` holds mutable or lazily-initialized state.
// [spec:ronin:req:make.no-ambient-state/test]
#[test]
fn test_no_mutable_process_globals() {
    let found = all_statics();
    let violations: Vec<&StaticItem> = found.iter().filter(|s| s.exemption.is_none()).collect();

    if !violations.is_empty() {
        let mut report = String::from(
            "mutable process globals found under kati/src-rs.\n\
             Evaluation state belongs on the session; see \
             plan/decisions/session-owned-evaluation.md.\n\
             A genuinely read-only item may be kept with a \
             `// no-globals-gate: <reason>` comment above it.\n\n",
        );
        for v in &violations {
            report.push_str(&format!("  src-rs/{}:{}: {}", v.file, v.line, v.name));
            if let Some((_, node)) = FORMER_OWNERS.iter().find(|(n, _)| *n == v.name) {
                report.push_str(&format!("  (removed by node {node}; it is back)"));
            }
            report.push('\n');
        }
        panic!("{report}");
    }
}

/// The two statics that are permitted are the two that are meant to be, and
/// each says why. A new exemption has to be added here as well as in the
/// source, so it cannot be granted quietly.
// [spec:ronin:req:make.no-ambient-state/test]
#[test]
fn test_permitted_statics_are_the_expected_ones() {
    let permitted: Vec<(String, String)> = all_statics()
        .into_iter()
        .filter_map(|s| s.exemption.map(|r| (s.name, r)))
        .collect();
    let names: Vec<&str> = permitted.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec!["FUNC_INFO_MAP", "GLOBAL"],
        "the set of permitted statics changed: {permitted:?}"
    );
    for (_, reason) in &permitted {
        assert!(!reason.is_empty(), "an exemption must say why");
    }
}

/// The scanner has to see a `static` inside a macro body, because that is the
/// shape that slips past a reviewer reading the macro definition rather than
/// its expansions — the statistics macros used to expand to one per call site.
#[test]
fn test_scan_sees_statics_inside_macro_bodies() {
    let source = r#"
macro_rules! collect_stats {
    ($name:literal) => {
        static STATS: std::sync::LazyLock<Stats> = std::sync::LazyLock::new(|| Stats::new($name));
    };
}
"#;
    let found = scan("example.rs", source);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].name, "STATS");
    assert_eq!(found[0].line, 4);
    assert!(found[0].exemption.is_none());
}

/// What the scanner must not report: `const` items and `&'static` lifetimes.
#[test]
fn test_scan_ignores_consts_and_static_lifetimes() {
    let source = r#"
const FUNC_INFO: &[FuncInfo] = &[];
pub const BOLD: &str = "\x1b[1m";
fn f(name: &'static str) -> &'static [u8] { name.as_bytes() }
/// A static would hold an index into whichever interner touched it first.
struct S { shellflag: &'static [u8] }
"#;
    assert_eq!(scan("example.rs", source), vec![]);
}

/// An exemption is recognised through an attribute, and its reason is kept.
#[test]
fn test_scan_reads_exemption_markers() {
    let source = r#"
// no-globals-gate: the global allocator, a Ronin-level choice
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
"#;
    let found = scan("example.rs", source);
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0].exemption.as_deref(),
        Some("the global allocator, a Ronin-level choice")
    );
}
