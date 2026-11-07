# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A benchmark of wildfire detection (smoke / fire / person) on the RGBT-3M UAV dataset using the Anthropic API. Two methods are implemented in-repo — a single-shot prompting baseline and an agentic loop where Claude writes CV code in a code-execution container and views its own generated images — and both are scored by one shared evaluator. There is no test suite, no linter config, and no packaging; everything runs as `python3 -m src.<module>`.

## Setup

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env      # loaded via python-dotenv, gitignored
```

Dataset must be unpacked into `rgbt-3m/` (gitignored except `data.yaml`), laid out as
`rgb/{train,test}/*.jpg`, `thermal/{train,test}/*.jpg`, `labels/{train,test}/*.txt`
(YOLO normalised cxcywh). Verify with:

```bash
python3 -c "from src.data_utils import load_frames; load_frames()"
```

## Commands

```bash
# single random frame, full trace + matplotlib GT-vs-pred plot
python3 -m src.agentic.main
python3 -m src.basic_prompting.main

# batch: start small — the agentic path costs ~$1/frame
python3 -m src.evaluation.run_benchmark agentic --videos video2 --limit 5 --workers 4
python3 -m src.evaluation.run_benchmark basic   --videos video2 --workers 32

# re-score what's on disk without re-paying for inference
python3 -m src.evaluation.run_benchmark agentic --score-only
```

Note: the module docstring in `run_benchmark.py` says `src.run_benchmark`; the real path is `src.evaluation.run_benchmark`.

## Architecture

**Shared core** — `src/constants.py` (class names, fixed 640x480), `src/schema.py` (the Pydantic `DetectionResult` every method must return, used as `output_format` for structured output), `src/data_utils.py` (frame discovery, GT loading, box drawing, raw-JPEG base64 encoding that deliberately avoids a second lossy re-encode).

**Method packages** are interchangeable behind one contract: `fn(client, stem, split) -> (detections, turns)`, wired up in `run_benchmark.get_detect`. `turns` is a list of per-turn usage dicts consumed by `src/evaluation/cost.py`. Each package owns its own `constants.py` (model, max tokens) and `prompt.py`.

**The agentic loop** (`src/agentic/utils.py:run`) is the non-obvious part. Claude *cannot see files its own container produces* — they come back as `file_id`s. The bridge:

1. `build_messages` uploads RGB + thermal via the Files API, attaches them both as inline base64 images *and* as `container_upload` blocks, and cache-controls the prompt (images + prompt are re-sent every turn).
2. Each turn, `harvest` scans `bash_code_execution_tool_result` blocks and records `{filename: file_id}` in a registry.
3. `attach_image` is a **client-side** tool (`src/agentic/tools.py`) that looks up the registry and returns an image block — this is what puts a container-generated PNG into Claude's context. The prompt requires it be called in a turn *separate* from the bash command that wrote the file.
4. `stop_reason == "pause_turn"` means continue the loop without sending tool results.
5. `commit` makes a final call with **no tools** and `output_format=DetectionResult` to get validated boxes.
6. `finally: cleanup` deletes uploads and every harvested file id. This matters — uploads and container captures persist workspace-wide and accumulate ~10/frame. Track ids in a `captured` list, not just the registry: reusing a filename overwrites its registry entry and orphans the earlier id.

**Evaluation** (`src/evaluation/metrics.py`) reports two numbers per class: standard mAP and *centre-coverage* (is the GT centre inside a correct-class prediction). They diverge sharply — fire is ~0.78 centre-coverage against AP ~0.05, which is a localization failure, not a detection failure.

Invariants that keep the comparison valid across methods (including the external SAM3 baseline):
- `MAX_DETS = [1, 10, 300]` — torchmetrics' default `[1, 10, 100]` silently drops the lowest-scoring excess and biases AP down. Every method must use the same cap.
- `_fit` orders corners *before* clamping, and is applied to GT as well as predictions.
- Overriding `max_detection_thresholds` renames the recall keys: `mar_300` exists, `mar_100` does not.

**Batch runner** (`src/evaluation/run_benchmark.py`) writes one JSON per frame to `results/<method>/frames/` and skips existing ones on rerun — so delete that directory after changing a prompt, or you will score stale outputs. `select()` is the single source of truth for which frames both inference and scoring touch. Concurrency uses a semaphore released by a `ramp` thread because rate limits are enforced sub-minute, so launching all workers at once trips 429s even at a fine per-minute average.

## Conventions

- Model is `claude-opus-5` with `thinking={"type": "adaptive"}` and `output_config={"effort": "high"}`; the Files API needs the `files-api-2025-04-14` beta, passed via `betas=BETAS`.
- Pricing constants in `cost.py` are Opus 5 USD/Mtok. `usage.input_tokens` already excludes cache reads — don't subtract them.
- Every detection carries a `description` field explaining the visual evidence; the prompts require it, and it's part of the project's story.
- Comments here explain *why* a non-obvious choice was made (rate-limit ramping, cleanup ordering, detection caps). Match that register rather than narrating what the code does.


## Secrets
Never read `.env` or key files. If you need a config value, ask.