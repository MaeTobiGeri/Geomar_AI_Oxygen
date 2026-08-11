# Code Style & Structure

Rules for how the rewrite (`BUILD_PLAN.md`) should actually be written, not just what it should do.
Applies to every file created under `BUILD_PLAN.md`'s phases. Five rules, in priority order when
they conflict with each other:

1. **Human-readable first.** Anyone should be able to open a file and understand what it does
   without needing this doc set open next to it. Prefer a longer, obvious name or an extra line of
   code over a clever one-liner. If a reader would have to pause and mentally trace through
   something, rewrite it plainer rather than adding a comment to explain it.
2. **As simple as possible.** Build what `SPEC.md` actually asks for, the plainest way that works.
   No frameworks, config layers, plugin systems, or abstraction introduced because it might be
   useful later — `BUILD_PLAN.md`'s phases are already scoped to what's needed now. If a function
   can be a function, it doesn't need to be a class. If a script can be a script, it doesn't need a
   CLI framework.
3. **No unnecessary checks.** Validate at the actual boundaries — raw file parsing (Phase 2, the
   data really is inconsistent between the two source files), the DWD API response, user input in
   the dashboard (Phase 10). Don't add `if x is None` guards, `try/except` blocks, or type checks
   for states that can't occur given how a function is actually called internally. Trust that
   `pandas`/`torch`/`pytorch-forecasting` do what their own contracts say; don't re-check their
   outputs "just in case."
4. **Only the code that's needed.** No speculative parameters, feature flags, or generalized
   versions of a function that only ever gets called one way. `SPEC.md` deliberately leaves several
   things as "decide once you see the data" rather than building configurability for every possible
   choice up front — implement the decision that was actually made, not a switch between every
   option that was considered. Delete code the moment it stops being used; don't comment it out
   "in case."
5. **Split by responsibility, not by file-size.** Never let unrelated concerns pile into one file —
   `BUILD_PLAN.md`'s suggested layout (`src/data_ingestion.py`, `src/pipeline.py`,
   `src/labeling.py`, `src/features.py`, `src/dataset.py`, `src/model.py`, plus top-level
   `train.py`/`evaluate.py`/`app.py`) is the intended split, one clear responsibility per file. If a
   file is doing two distinct jobs (e.g. data cleaning and feature engineering), split it even if
   both halves are individually short. Conversely, don't split a single cohesive piece of logic
   across files just to keep files small — the boundary is "what it does," not line count.

## What this looks like in practice

**Comments**: none by default. Add one only when the *why* isn't obvious from the code itself — a
non-obvious physical/statistical reason for a specific number or transform (`SPEC.md` already
documents most of these — link to the relevant `SPEC.md` section instead of re-explaining the reason
inline), a workaround for a specific library quirk, or an invariant a future edit could easily
break. Never a comment that restates what the next line already says.

**Error handling**: a data-loading function that expects three specific files with a specific
header-row count doesn't need to handle "file has a different number of header rows" — that's not a
real scenario for this pipeline, it's not a public library. A dashboard input field that takes a
user-picked date *does* need to handle "no checkpoint has been trained yet" (`SPEC.md` §10 already
calls this out) — that's a real, reachable state. The test is always "can this actually happen given
how this code is actually called," not "could this theoretically happen in some other context."

**Abstraction**: `SPEC.md` §5 asks for a forward feature-selection *procedure* — write it as a plain
function that does the selection, not a generic pluggable "selection strategy" interface. There's
one model (`SPEC.md` §7 recommends a TFT with a documented fallback, not both built simultaneously)
— don't build a model-agnostic wrapper layer until/unless a second model architecture actually gets
built.

**File structure**: `BUILD_PLAN.md`'s layout is the default — treat a change to it (splitting a file
further, merging two) as a deliberate call tied to a specific readability problem, not a
reorganization for its own sake. A phase's output should be legible on its own: someone reading
`src/labeling.py` shouldn't need to also read `src/dataset.py` to understand what it does.

**Tests**: `BUILD_PLAN.md` Phase 11 already scopes testing to what actually matters for this
project — the two easy-to-silently-break data steps (unit conversion, imputation limits) and a
smoke test that the app doesn't crash. That's deliberate, not a starting point to expand from by
default — more tests are worth adding when a specific bug shows they're needed, not preemptively.
