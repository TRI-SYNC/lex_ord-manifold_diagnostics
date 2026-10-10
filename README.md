# lex_ord-manifold_diagnostics

Two small, independently verifiable code artifacts, extracted and published as a standalone
reference point: both are real, tested, and run as described — unlike a number of other
AI-generated "technical" documents checked against them, which were not.

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

## Why this repo exists

Both artifacts were checked against claims made by an AI chatbot in the course of
sorting a large set of AI-generated documents into "real / implemented" versus
"fabricated / never implemented." These two passed every check: the described behavior
matches the real implementation, and the implementation's own test suite passes. They're
published here as a plain, checkable reference — clone it, run the tests yourself.
