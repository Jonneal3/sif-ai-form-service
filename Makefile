PYTHON?=python3
REFRESH_CMD=$(PYTHON) scripts/refresh_feedback_pipeline.py \
  --cases-out eval/feedback_cases.jsonl \
  --failures-out eval/feedback_failures.jsonl \
  --optimize-out examples/next_steps_examples.optimized.jsonl \
  --include-negative

.PHONY: refresh-feedback
refresh-feedback:
	$(REFRESH_CMD)
