PYTHON?=python3
PYTHONPATH?=src
REFRESH_CMD=$(PYTHON) scripts/refresh_feedback_pipeline.py \
  --cases-out eval/feedback_cases.jsonl \
  --failures-out eval/feedback_failures.jsonl \
  --optimize-out examples/next_steps_examples.optimized.jsonl \
  --include-negative

.PHONY: refresh-feedback
refresh-feedback:
	PYTHONPATH=$(PYTHONPATH) $(REFRESH_CMD)

INSIGHTS_LIMIT?=2000
OPTIMIZER_ARCHIVE_DIR?=data/archives/optimizer_runs

.PHONY: refresh-feedback-insights
refresh-feedback-insights:
	PYTHONPATH=$(PYTHONPATH) $(REFRESH_CMD) --collect-insights --insights-limit $(INSIGHTS_LIMIT) --archive-dir $(OPTIMIZER_ARCHIVE_DIR)

.PHONY: telemetry-insights
telemetry-insights:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/telemetry_insights.py \
	  --checkpoint .telemetry_checkpoint.json \
	  --summary data/telemetry_summary.json \
	  --limit $(INSIGHTS_LIMIT)

COMPILE_DATASET?=eval/datasets/next_steps_structural.jsonl
COMPILE_OUT?=compiled/next_steps_compiled.jsonl
COMPILE_ARCHIVE_DIR?=data/archives/compiled_runs

.PHONY: compile-next-steps
compile-next-steps:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) compile/compile_next_steps.py --dataset $(COMPILE_DATASET) --output $(COMPILE_OUT) --archive-dir $(COMPILE_ARCHIVE_DIR)

.PHONY: show-last-optimizer-run
show-last-optimizer-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/show_last_optimizer_run.py
