# Repository Guidelines

## Project Structure & Module Organization

This repository centers on a multimodal League of Legends video QA pipeline. Core Python modules are organized by pipeline phase:

* `knowledge_extraction/`: video splitting, ASR, VLM frame descriptions, champion/entity matching, and segment summaries.
* `knowledge_sanitization/`: pre-build and post-build cleaning, reports, quarantine logs, and sanitized cache generation.
* `knowledge_build/`: chunking, vector indexes, knowledge graph construction, and global graph merge.
* `knowledge_inference/`: retrieval, reranking, prompt context building, answer generation, and confidence/debug output.
* `knowledge_pipeline/`: queue orchestration for the full workflow.
* `knowledge_api_server/`: FastAPI inference API.
* `knowledge_frontend/`: Next.js chat UI.
* `docs/`: architecture diagrams and phase documentation.
* `qa_datasets/` and `knowledge_system_evaluation/`: evaluation datasets and reporting scripts.
* `knowledge_system_evaluation_v2/`: current active workspace for the IEEE Access paper evaluation.

Generated artifacts such as `knowledge_build_cache_*`, `knowledge_sanitization/cache/`, `downloads/`, model folders, `.next/`, `node_modules/`, and `venv_*` should be treated as local outputs, not primary source.

## Development Commands and Manual High-Impact Operations

The full pipeline, queue runners, extraction, build, sanitization, cache rebuilds, and evaluation commands below are manual operations. Do not run them automatically as part of exploration, implementation, or routine validation. Explicit approval under Rule 4 is required before running them.

Run the full pipeline dry run:

```bash
source venv_smolvlm/bin/activate
python -m knowledge_pipeline.run_full_queue --dry-run
```

Run the full queue:

```bash
python -m knowledge_pipeline.run_full_queue
```

Query inference locally:

```bash
python -m knowledge_inference.cli --query "What happened around the first dragon fight?" --debug
```

Start the API:

```bash
uvicorn knowledge_api_server.main:app --host 0.0.0.0 --port 8000
```

Frontend commands:

```bash
cd knowledge_frontend
npm install
npm run dev
npm run build
npm run lint
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, type hints where practical, `snake_case` for modules/functions, and `PascalCase` for classes. Keep pipeline code phase-specific; avoid cross-phase shortcuts that bypass sanitized artifacts. Frontend code uses TypeScript, React, Next.js, and ESLint.

# Coding Agent Rules

## 1. Default to analysis, not implementation

Unless the user explicitly asks to **implement, modify, apply, fix, refactor, create, or edit** something, operate in analysis-only mode.

Analysis-only mode may include:

* Reading relevant code and documentation.
* Explaining behaviour, risks, root causes, and trade-offs.
* Proposing a minimal patch plan.
* Showing a suggested diff or code snippet without applying it.

Questions such as “why does this happen?”, “can you review this?”, “what should we do?”, or “is this correct?” do **not** authorize file edits.

A vague request such as “make this better,” “improve it,” or “fix the pipeline” authorizes analysis and a proposed plan only, unless the requested target, expected behaviour, and affected area are reasonably clear.

A completed explanation, diagnosis, plan, or suggested patch is a valid final answer. Do not convert analysis into implementation unless the user explicitly asks for implementation.

When the request is ambiguous, do not edit code. State the recommended change, affected files, and smallest validation step.

## 2. Lock the scope before changing anything

Before implementation, state:

1. The exact requested outcome.
2. What is explicitly out of scope.
3. The smallest set of files expected to change.
4. The narrowest validation command.

Treat the user’s request as one task. Do not add cleanup, refactors, dependency upgrades, renaming, formatting sweeps, documentation rewrites, new abstractions, or related improvements unless explicitly requested.

If you discover adjacent problems, report them separately but do not fix them in the same change.

## 3. Make the smallest reversible change

Prefer a focused patch over a broad improvement.

* Modify only files necessary for the requested behaviour.
* Preserve existing interfaces, filenames, schemas, cache formats, and pipeline contracts unless the user explicitly asks to change them.
* Do not replace working logic with a “cleaner” design unless the current design blocks the requested task.
* Do not create helper layers, config systems, abstractions, or tests merely because they may be useful later.
* Do not make unrelated formatting, linting, naming, or documentation changes.

Stop once the requested success criterion is met.

## 4. Require approval for high-impact operations

Do not perform any of the following without explicitly telling the user what will happen and receiving a clear go-ahead:

* Running the full pipeline, any queue runner, extraction, build, sanitization, or evaluation batch.
* Running a full-pipeline dry run.
* Using `--force`.
* Downloading models, installing packages, creating environments, or changing model/runtime configuration.
* Rebuilding vector databases, graphs, caches, or global artifacts.
* Deleting, overwriting, moving, or regenerating existing reports, caches, datasets, or evaluation outputs.
* Changing evaluation datasets, gold answers, methodology files, metrics, ablation definitions, or dataset splits.
* Database migrations, schema changes, dependency upgrades, lockfile updates, or API contract changes.
* Destructive Git operations, including `reset`, `clean`, forced checkout, rebase, history rewriting, or force-push.

For these operations, first provide the exact command, affected paths, expected generated artifacts, and likely cost or side effects.

## 5. Treat generated artifacts as protected runtime inputs

Although caches and reports are generated artifacts, do not treat them as disposable.

In particular, do not casually modify or delete:

* `knowledge_sanitization/cache/sanitized_*`
* `knowledge_build_cache_*`
* `knowledge_build_cache_global`
* vector databases and GraphML files
* evaluation datasets and generated evaluation reports

Use a new, clearly named output path for experiments whenever possible. Never overwrite an existing evaluation result or cache unless the user explicitly identifies it as safe to replace.

## 6. Preserve repository state

Before editing, inspect the relevant working-tree state.

* Do not modify, discard, stage, stash, reset, or reformat unrelated user changes.
* Do not touch files outside the requested scope.
* After editing, inspect the final diff and ensure it contains only the intended change.
* Do not create commits, branches, pull requests, or push changes unless explicitly asked.

## 7. Validate proportionally

Only validate changes that were actually made.

* Use the narrowest relevant, low-impact check first.
* Prefer targeted unit tests, syntax checks, static checks, `npm run lint`, or a narrowly scoped inference CLI query when it does not alter artifacts.
* Do not run the full pipeline merely to validate a local change.
* Do not run a batch evaluation merely to validate one implementation detail.
* Add a new test only when the requested behaviour changes and an existing targeted test cannot cover it.

Full-pipeline dry runs, queue runners, extraction, build, sanitization, cache rebuilds, and evaluation scripts are high-impact operations and require explicit approval under Rule 4 before execution.

If validation fails for an unrelated reason, report the failure and stop. Do not expand the task into fixing unrelated failures.

## 8. Report concisely and stop

After implementation, report only:

* Files changed.
* Behaviour changed.
* Validation actually run and its result.
* Anything intentionally not validated.

Do not continue with optional improvements after the requested task is complete. Do not create follow-up tasks, refactors, or “nice-to-have” changes without an explicit new request.

## Testing Guidelines

There is no single repository-wide test runner.

Use the narrowest relevant, low-impact validation command for the requested change. Prefer targeted unit tests, syntax checks, static checks, frontend linting, or a narrowly scoped inference CLI query that does not alter artifacts.

Do not run pipeline dry runs, queue runners, extraction, build, sanitization, cache rebuilds, or evaluation scripts without explicit approval under Rule 4.

Create or name new evaluation files only when the user explicitly requests an evaluation artifact. Follow existing naming patterns such as `qa_datasets/*_QA_Eval.json` when creating approved evaluation files.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive, mostly lowercase summaries rather than strict Conventional Commits. Keep commits focused, for example:

```text
improve sanitization quality gate
update inference reranker
```

Do not create commits, branches, pull requests, or push changes unless explicitly requested.

When explicitly asked to prepare a pull request, include:

* A short problem statement.
* A summary of changed pipeline phases.
* Commands actually run.
* Generated artifact impact.
* Screenshots for frontend changes.
* Required models, `.env` values, or cache rebuilds.

## Security & Configuration Tips

Do not commit secrets, local `.env` values, model weights, downloaded videos, or regenerated caches unless explicitly requested.

Inference should read from `knowledge_sanitization/cache/` only. Preserve that boundary when adding features.

Do not alter environment configuration, model selection, embedding models, runtime dependencies, or virtual environments without explicit approval under Rule 4.

## Current Active Evaluation (IEEE Access)

We are currently executing an evaluation of the system for a paper publication in **IEEE Access**.

The evaluation methodology, including dataset splitting, legacy tagging, ablation testing (Graph-RAG vs Vector-only vs BM25), and LLM-as-a-judge scoring, is defined in:

* `knowledge_system_evaluation_v2/EVALUATION_PLAN.md`
* `knowledge_system_evaluation_v2/final_evaluation_integration.md`

When assisting with this evaluation, strictly adhere to this methodology, especially the critical separation between *System Accuracy* and *Knowledge Base Currency*.

Treat evaluation datasets, gold answers, methodology documents, split definitions, ablation settings, scoring scripts, and generated reports as read-only unless the user explicitly requests a change. Do not silently alter them to make results look better, simplify an experiment, or resolve an inconsistency.
