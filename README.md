# lex_ord-manifold_diagnostics

Two small, independently verifiable code artifacts: `lex_ord` (Rust) and
`manifold_diagnostics` (Python). Clone the repo and run the test suites yourself to check
the claims below.

## What's here

### `rust/lex_ord`
Deterministic, locale-independent lexicographic ordering over UTF-8 byte sequences.
Extracted from a real consensus-engine codebase, where it is used for canonical JSON
serialization and other places requiring reproducible ordering across machines.

```
cd rust/lex_ord
cargo test
```
27 tests, covering uppercase/lowercase ordering, numeric-string ordering, prefix rules,
non-ASCII UTF-8 byte ordering, leading zeros, whitespace, empty strings, control bytes,
and a `BTreeMap`-backed `LexStr` newtype.

### `python/manifold_diagnostics`
Generic numerical diagnostics: finite-difference Jacobians and power-iteration singular
values (sensitivity analysis), subspace projection distance (on/off-manifold / anomaly
detection), nearest-prototype manifold identification with confidence scoring, linear
margin computation, and a deterministic SHA-256 fingerprint of a matrix's dominant
singular-vector subspace (sign-corrected to remove SVD's inherent sign ambiguity).

```
cd python/manifold_diagnostics
pip install -e .
pytest
```
4 tests, covering Jacobian correctness, power-iteration convergence to a known singular
value, hash stability under repeated runs, and zero-distance projection onto an identity
subspace.
