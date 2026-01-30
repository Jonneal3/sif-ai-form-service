PYTHON?=python3
PYTHONPATH?=.
PLANNER_OPT_DIR?=src/programs/question_planner/data/optimized_outputs
PLANNER_RUNS_DIR?=$(PLANNER_OPT_DIR)/runs
PLANNER_RUN_ID?=$(shell date +%Y-%m-%d_%H-%M-%S)
# Freeze the run id for the duration of this `make` invocation (avoid timestamp drift across steps).
PLANNER_RUN_ID:=$(PLANNER_RUN_ID)
PLANNER_RUN_DIR:=$(PLANNER_RUNS_DIR)/$(PLANNER_RUN_ID)
PLANNER_MAX_TOKENS?=3200
PLANNER_TEMPERATURE?=0.3
PLANNER_METRIC_MAX_TOKENS?=1200
PLANNER_NUM_THREADS?=1
PLANNER_NUM_CANDIDATE_PROGRAMS?=6
PLANNER_TRAINSET_LIMIT?=0
PLANNER_TRAINSET_SHUFFLE?=false

.PHONY: dev
dev:
	PYTHONPATH=.:src $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8008 --reload

.PHONY: optimize-planner-core
optimize-planner-core:
	@mkdir -p $(PLANNER_RUN_DIR)
	AI_FORM_TOKEN_TELEMETRY=true DSPY_TRACK_USAGE=true \
	DSPY_PLANNER_MAX_TOKENS=$(PLANNER_MAX_TOKENS) DSPY_PLANNER_TEMPERATURE=$(PLANNER_TEMPERATURE) \
	DSPY_PLANNER_METRIC_MAX_TOKENS=$(PLANNER_METRIC_MAX_TOKENS) \
	PYTHONPATH=.:src $(PYTHON) -m optimizers.optimize_question_planner \
		--num-threads $(PLANNER_NUM_THREADS) \
		--num-candidate-programs $(PLANNER_NUM_CANDIDATE_PROGRAMS) \
		$(if $(filter true,$(PLANNER_TRAINSET_SHUFFLE)),--trainset-shuffle,) \
		$(if $(filter-out 0,$(PLANNER_TRAINSET_LIMIT)),--trainset-limit $(PLANNER_TRAINSET_LIMIT),) \
		--out $(PLANNER_RUN_DIR)/question_planner_optimized.json \
		--export-demo-pack $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl \
		--usage-report $(PLANNER_RUN_DIR)/token_usage.json
	@echo "ok: wrote run -> $(PLANNER_RUN_DIR)"

.PHONY: optimize-planner
optimize-planner: optimize-planner-core export-planner-demos-pretty-run export-planner-demos-json-pretty-run report-planner-demos-run
	@echo ""
	@echo "Report:"
	@cat $(PLANNER_RUN_DIR)/question_planner_demo_pack.report.txt

.PHONY: optimize
optimize: check-planner-examples optimize-planner

.PHONY: optimize-fast
optimize-fast:
	$(MAKE) optimize \
		PLANNER_NUM_CANDIDATE_PROGRAMS=2 \
		PLANNER_TRAINSET_LIMIT=5 \
		PLANNER_TRAINSET_SHUFFLE=true

.PHONY: optimize-planner-all
optimize-planner-all: optimize-planner

.PHONY: use-optimized-planner-demos
use-optimized-planner-demos:
	@test -f $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl || ( \
		echo "Not found: $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl"; \
		echo "Pass PLANNER_RUN_ID=<run folder name under $(PLANNER_RUNS_DIR)>"; \
		exit 1; \
	)
	@echo export DSPY_PLANNER_DEMO_PACK=$(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl

.PHONY: inspect-planner-demos
inspect-planner-demos:
	@test -f $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl || ( \
		echo "Not found: $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl"; \
		echo "Pass PLANNER_RUN_ID=<run folder name under $(PLANNER_RUNS_DIR)>"; \
		exit 1; \
	)
	PYTHONPATH=.:src $(PYTHON) scripts/inspect_question_planner_demos.py $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl --format plain

.PHONY: export-planner-demos-pretty
export-planner-demos-pretty:
	@test -f $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl || ( \
		echo "Not found: $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl"; \
		echo "Pass PLANNER_RUN_ID=<run folder name under $(PLANNER_RUNS_DIR)>"; \
		exit 1; \
	)
	PYTHONPATH=.:src $(PYTHON) scripts/inspect_question_planner_demos.py \
		$(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl \
		--format plain \
		--out $(PLANNER_RUN_DIR)/question_planner_demo_pack.pretty.txt

.PHONY: export-planner-demos-pretty-run
export-planner-demos-pretty-run:
	PYTHONPATH=.:src $(PYTHON) scripts/inspect_question_planner_demos.py \
		$(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl \
		--format plain \
		--out $(PLANNER_RUN_DIR)/question_planner_demo_pack.pretty.txt

.PHONY: export-planner-demos-json-pretty
export-planner-demos-json-pretty:
	@test -f $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl || ( \
		echo "Not found: $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl"; \
		echo "Pass PLANNER_RUN_ID=<run folder name under $(PLANNER_RUNS_DIR)>"; \
		exit 1; \
	)
	PYTHONPATH=.:src $(PYTHON) scripts/inspect_question_planner_demos.py \
		$(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl \
		--format json \
		--out $(PLANNER_RUN_DIR)/question_planner_demo_pack.pretty.json.txt

.PHONY: export-planner-demos-json-pretty-run
export-planner-demos-json-pretty-run:
	PYTHONPATH=.:src $(PYTHON) scripts/inspect_question_planner_demos.py \
		$(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl \
		--format json \
		--out $(PLANNER_RUN_DIR)/question_planner_demo_pack.pretty.json.txt

.PHONY: report-planner-demos
report-planner-demos:
	@test -f $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl || ( \
		echo "Not found: $(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl"; \
		echo "Pass PLANNER_RUN_ID=<run folder name under $(PLANNER_RUNS_DIR)>"; \
		exit 1; \
	)
	DSPY_PLANNER_METRIC_MAX_TOKENS=$(PLANNER_METRIC_MAX_TOKENS) \
	DSPY_PLANNER_METRIC_TEMPERATURE=0.0 \
	PYTHONPATH=.:src $(PYTHON) scripts/report_question_planner_scores.py \
		$(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl \
		--out $(PLANNER_RUN_DIR)/question_planner_demo_pack.report.txt

.PHONY: report-planner-demos-run
report-planner-demos-run:
	DSPY_PLANNER_METRIC_MAX_TOKENS=$(PLANNER_METRIC_MAX_TOKENS) \
	DSPY_PLANNER_METRIC_TEMPERATURE=0.0 \
	PYTHONPATH=.:src $(PYTHON) scripts/report_question_planner_scores.py \
		$(PLANNER_RUN_DIR)/question_planner_demo_pack.jsonl \
		--out $(PLANNER_RUN_DIR)/question_planner_demo_pack.report.txt

.PHONY: check-planner-examples
check-planner-examples:
	PYTHONPATH=.:src $(PYTHON) scripts/validate_question_planner_examples.py

.PHONY: export-openapi-contract
export-openapi-contract:
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/export_openapi_contract.py --out api/api-contract/openapi.json

.PHONY: verify-openapi-contract
verify-openapi-contract:
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/verify_openapi_contract.py --contract api/api-contract/openapi.json

.PHONY: check-example-leaks
check-example-leaks:
	PYTHONPATH=.:src:$(PYTHONPATH) $(PYTHON) scripts/check_example_leaks.py

.PHONY: check-service-openapi-contract
check-service-openapi-contract:
	PYTHONPATH=.:src:$(PYTHONPATH) $(PYTHON) scripts/check_service_openapi_contract.py
