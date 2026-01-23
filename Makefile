PYTHON?=python3
PYTHONPATH?=.

.PHONY: dev
dev:
	PYTHONPATH=.:src $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8008 --reload

.PHONY: export-openapi-contract
export-openapi-contract:
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/export_openapi_contract.py --out api/api-contract/openapi.json

.PHONY: verify-openapi-contract
verify-openapi-contract:
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/verify_openapi_contract.py --contract api/api-contract/openapi.json

.PHONY: check-example-leaks
check-example-leaks:
	PYTHONPATH=.:src:$(PYTHONPATH) $(PYTHON) scripts/check_example_leaks.py

.PHONY: check-batchgen-examples
check-batchgen-examples:
	PYTHONPATH=.:src:$(PYTHONPATH) $(PYTHON) scripts/check_batchgen_examples.py
