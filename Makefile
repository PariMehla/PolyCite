.PHONY: install test reproduce reproduce-live clean

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest tests/ -v

reproduce:
	PYTHONPATH=. POLYCITE_AUTO_CONFIRM=1 python scripts/run_pipeline.py --mode dry-run

reproduce-live:
	@test -n "$$COHERE_API_KEY" || (echo "COHERE_API_KEY is not set. Get a free trial key: https://dashboard.cohere.com/api-keys" && exit 1)
	PYTHONPATH=. python scripts/run_pipeline.py --mode live

clean:
	rm -rf cache/responses cache/call_count.json cache/dry-run results/*.parquet figures/*.png
