PYTHON?=python3
PYTHONPATH?=.

.PHONY: export-openapi-contract
export-openapi-contract:
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/export_openapi_contract.py --out api/api-contract/openapi.json

.PHONY: verify-openapi-contract
verify-openapi-contract:
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/verify_openapi_contract.py --contract api/api-contract/openapi.json
