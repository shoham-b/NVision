---
trigger: always_on
---

# Guardrails: no unintended actions

- **Stay in scope.** Change only what the task asks for. No drive-by refactors, renames, reformatting of untouched files, or dependency/`pyproject.toml`/CI changes unless requested. If you spot something else worth fixing, say so instead of doing it.
- **Never touch the shared cache or results.** `artifacts/` (incl. `artifacts/cache/`, tens of GB and shared across worktrees), `data/`, `logs/` are read-only for you. No `nv cache clean`, no deleting or overwriting cache/DB files, no full `nv run` / `nv groups run`. Use `--dry-run` or `run-single --repeats 1 --dry-run`.
- **No git side effects unasked.** Do not commit, push, force-push, reset --hard, checkout over changes, stash, or rebase unless the user asks. Never use `--no-verify`.
- **No destructive or outward-facing commands** (rm -r, drop/truncate, kill, deploys, GCP/docker pushes, anything under `.github/workflows` or `scripts/` that talks to a remote) without explicit confirmation in chat.
- **Physics/algorithm changes need a heads-up.** If you change SMC, SBED, likelihood, focus, or a physics constant, state whether old results are stale (re-run vs `nv render`) and whether the physics fingerprint must be bumped. Do not bump it silently.
- **Fail fast, no silent fallbacks.** Raise on shape/coordinate mismatches; do not add try/except, clipping, or defaults that hide a bug. Keep `_unit` / `_phys` suffixes strict.
- **Scratch only in `scratch/` and `debug/`.** Never in the repo root.
- **Uncertain or ambiguous? Ask first**, with a recommendation, rather than choosing a risky interpretation.
