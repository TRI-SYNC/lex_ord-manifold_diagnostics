//! Lexicographic ordering over UTF-8 byte sequences.
//!
//! Rules (frozen):
//! - Ordering is byte-based, not locale-aware, and not Unicode-collation-aware.
//! - Case-sensitive: uppercase bytes sort before lowercase ("A" < "a").
//! - Numeric strings sort lexicographically, not numerically ("10" < "2").
//! - Prefix rule: a string that is a prefix of another sorts first ("abc" < "abcd").
//! - Non-ASCII characters sort by UTF-8 byte value ("é" > "z" because 0xC3 > 0x7A).
//! - ASCII always sorts before any multi-byte UTF-8 sequence.
//! - Leading zeros are significant ("001" < "01" < "1").
//! - Whitespace sorts before visible ASCII (" a" < "a").
//! - Empty string sorts before every non-empty string.
//! - Control bytes sort by raw byte value.
//! - No locale rules. No Unicode normalisation. No Unicode collation algorithm.
//!
//! This ordering is required for deterministic hashing, replay, and
//! reviewer-safe reproducibility across all engine components.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Compare two strings using lexicographic UTF-8 byte ordering.
///
/// This is the canonical comparator for all string fields in the engine.
/// It is equivalent to `a.as_bytes().cmp(b.as_bytes())`.
#[inline]
pub fn lex_cmp(a: &str, b: &str) -> Ordering {
    a.as_bytes().cmp(b.as_bytes())
}

/// Sort a slice of `String` values in-place using lexicographic UTF-8 byte order.
pub fn sort_lex(items: &mut [String]) {
    items.sort_by(|a, b| lex_cmp(a, b));
}

/// Sort a slice of `T` in-place by a `String` key using lexicographic UTF-8 byte order.
///
/// `key_fn` is called twice per comparison. Prefer small key strings.
pub fn sort_lex_by_key<T, F>(items: &mut [T], key_fn: F)
where
    F: Fn(&T) -> String,
{
    items.sort_by(|a, b| lex_cmp(&key_fn(a), &key_fn(b)));
}

/// Return `true` if `items` is already sorted in lexicographic UTF-8 byte order
/// (non-descending: equal adjacent elements are allowed).
pub fn is_lex_sorted<S: AsRef<str>>(items: &[S]) -> bool {
    items
        .windows(2)
        .all(|w| lex_cmp(w[0].as_ref(), w[1].as_ref()) != Ordering::Greater)
}

/// A `String` newtype whose `Ord` implementation is explicitly bound to
/// lexicographic UTF-8 byte ordering.
///
/// Use `LexStr` as a key type in `BTreeMap` or any sorted collection where
/// the ordering must be frozen and reproducible.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LexStr(pub String);

impl LexStr {
    /// Wrap a `String`.
    pub fn new(s: impl Into<String>) -> Self {
        LexStr(s.into())
    }

    /// Borrow the inner `str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialOrd for LexStr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LexStr {
    fn cmp(&self, other: &Self) -> Ordering {
        lex_cmp(&self.0, &other.0)
    }
}

impl std::fmt::Display for LexStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for LexStr {
    fn from(s: String) -> Self {
        LexStr(s)
    }
}

impl From<&str> for LexStr {
    fn from(s: &str) -> Self {
        LexStr(s.to_owned())
    }
}

impl AsRef<str> for LexStr {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── lex_cmp edge cases ────────────────────────────────────────────────────

    #[test]
    fn uppercase_before_lowercase() {
        // 'A' = 0x41, 'a' = 0x61
        assert_eq!(lex_cmp("A", "a"), Ordering::Less);
        assert_eq!(lex_cmp("a", "A"), Ordering::Greater);
    }

    #[test]
    fn numeric_strings_lexicographic_not_numeric() {
        // "10" vs "2": '1' (0x31) < '2' (0x32) → "10" < "2"
        assert_eq!(lex_cmp("10", "2"), Ordering::Less);
        assert_eq!(lex_cmp("2", "10"), Ordering::Greater);
        assert_eq!(lex_cmp("9", "10"), Ordering::Greater);
    }

    #[test]
    fn prefix_rule_shorter_first() {
        // "abc" is a prefix of "abcd" → "abc" < "abcd"
        assert_eq!(lex_cmp("abc", "abcd"), Ordering::Less);
        assert_eq!(lex_cmp("abcd", "abc"), Ordering::Greater);
    }

    #[test]
    fn non_ascii_sorts_by_utf8_bytes() {
        // "é" = [0xC3, 0xA9], "z" = [0x7A] → "é" > "z"
        assert_eq!(lex_cmp("é", "z"), Ordering::Greater);
        assert_eq!(lex_cmp("z", "é"), Ordering::Less);
    }

    #[test]
    fn ascii_before_multibyte_utf8() {
        // Every ASCII byte is ≤ 0x7F; every multi-byte UTF-8 first byte is ≥ 0xC0
        assert_eq!(lex_cmp("z", "é"), Ordering::Less);
        assert_eq!(lex_cmp("~", "À"), Ordering::Less); // '~' = 0x7E, 'À' starts 0xC3
    }

    #[test]
    fn leading_zeros_significant() {
        // "001" < "01" < "1"
        assert_eq!(lex_cmp("001", "01"), Ordering::Less);
        assert_eq!(lex_cmp("01", "1"), Ordering::Less);
        assert_eq!(lex_cmp("001", "1"), Ordering::Less);
    }

    #[test]
    fn whitespace_before_visible_ascii() {
        // ' ' = 0x20, 'a' = 0x61 → " a" < "a"
        assert_eq!(lex_cmp(" a", "a"), Ordering::Less);
        assert_eq!(lex_cmp("a", " a"), Ordering::Greater);
    }

    #[test]
    fn empty_string_sorts_first() {
        assert_eq!(lex_cmp("", "a"), Ordering::Less);
        assert_eq!(lex_cmp("a", ""), Ordering::Greater);
        assert_eq!(lex_cmp("", ""), Ordering::Equal);
    }

    #[test]
    fn control_bytes_by_raw_value() {
        // '\x01' < '\x02' < '\t' (0x09) < ' ' (0x20)
        assert_eq!(lex_cmp("\x01", "\x02"), Ordering::Less);
        assert_eq!(lex_cmp("\x02", "\t"), Ordering::Less);
        assert_eq!(lex_cmp("\t", " "), Ordering::Less);
    }

    #[test]
    fn equal_strings_are_equal() {
        assert_eq!(lex_cmp("hello", "hello"), Ordering::Equal);
        assert_eq!(lex_cmp("", ""), Ordering::Equal);
    }

    // ── sort_lex ─────────────────────────────────────────────────────────────

    #[test]
    fn sort_lex_empty() {
        let mut items: Vec<String> = vec![];
        sort_lex(&mut items);
        assert!(items.is_empty());
    }

    #[test]
    fn sort_lex_single() {
        let mut items = vec!["a".to_owned()];
        sort_lex(&mut items);
        assert_eq!(items, ["a"]);
    }

    #[test]
    fn sort_lex_mixed() {
        let mut items = vec![
            "banana".to_owned(),
            "Apple".to_owned(),
            "10".to_owned(),
            "2".to_owned(),
            "".to_owned(),
            " a".to_owned(),
        ];
        sort_lex(&mut items);
        // '' < ' a' < '10' < '2' < 'Apple' < 'banana'
        assert_eq!(items, ["", " a", "10", "2", "Apple", "banana"]);
    }

    #[test]
    fn sort_lex_already_sorted_unchanged() {
        let mut items = vec!["A".to_owned(), "B".to_owned(), "a".to_owned()];
        sort_lex(&mut items);
        assert_eq!(items, ["A", "B", "a"]);
    }

    // ── sort_lex_by_key ───────────────────────────────────────────────────────

    #[test]
    fn sort_lex_by_key_orders_correctly() {
        let mut items = vec![
            ("banana".to_owned(), 1_i32),
            ("Apple".to_owned(), 2),
            ("10".to_owned(), 3),
        ];
        sort_lex_by_key(&mut items, |(s, _)| s.clone());
        assert_eq!(items[0].0, "10");
        assert_eq!(items[1].0, "Apple");
        assert_eq!(items[2].0, "banana");
    }

    // ── is_lex_sorted ─────────────────────────────────────────────────────────

    #[test]
    fn is_lex_sorted_true_for_sorted_strings() {
        assert!(is_lex_sorted(&["", "10", "2", "Apple", "banana"]));
    }

    #[test]
    fn is_lex_sorted_true_for_equal_elements() {
        assert!(is_lex_sorted(&["a", "a", "b"]));
    }

    #[test]
    fn is_lex_sorted_true_for_empty_slice() {
        let empty: &[&str] = &[];
        assert!(is_lex_sorted(empty));
    }

    #[test]
    fn is_lex_sorted_false_for_unsorted() {
        assert!(!is_lex_sorted(&["banana", "Apple"]));
        assert!(!is_lex_sorted(&["2", "10"]));
    }

    // ── LexStr ───────────────────────────────────────────────────────────────

    #[test]
    fn lex_str_uppercase_before_lowercase() {
        let a = LexStr::from("A");
        let b = LexStr::from("a");
        assert!(a < b);
    }

    #[test]
    fn lex_str_numeric_lexicographic() {
        let ten = LexStr::from("10");
        let two = LexStr::from("2");
        assert!(ten < two);
    }

    #[test]
    fn lex_str_prefix_rule() {
        let short = LexStr::from("abc");
        let long = LexStr::from("abcd");
        assert!(short < long);
    }

    #[test]
    fn lex_str_equal() {
        assert_eq!(LexStr::from("hello"), LexStr::from("hello"));
    }

    #[test]
    fn lex_str_in_btreemap_sorts_correctly() {
        use std::collections::BTreeMap;
        let mut map: BTreeMap<LexStr, i32> = BTreeMap::new();
        map.insert(LexStr::from("banana"), 1);
        map.insert(LexStr::from("Apple"), 2);
        map.insert(LexStr::from("10"), 3);
        map.insert(LexStr::from("2"), 4);

        let keys: Vec<&str> = map.keys().map(LexStr::as_str).collect();
        assert_eq!(keys, ["10", "2", "Apple", "banana"]);
    }

    #[test]
    fn lex_str_display() {
        assert_eq!(LexStr::from("hello").to_string(), "hello");
    }

    #[test]
    fn lex_str_new_and_as_str() {
        let s = LexStr::new("test");
        assert_eq!(s.as_str(), "test");
    }

    #[test]
    fn lex_str_as_ref_str() {
        let s = LexStr::from("hello");
        let r: &str = s.as_ref();
        assert_eq!(r, "hello");
    }
}
