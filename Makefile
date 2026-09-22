.PHONY: install test reproduce reproduce-live fertility clean

install:
	python3 -m pip install -r requirements.txt

test:
	PYTHONPATH=. python3 -m pytest tests/ -v

reproduce:
	PYTHONPATH=. POLYCITE_AUTO_CONFIRM=1 python3 scripts/run_pipeline.py --mode dry-run

reproduce-live:
	@test -n "$$COHERE_API_KEY" || (echo "COHERE_API_KEY is not set. Get a free trial key: https://dashboard.cohere.com/api-keys" && exit 1)
	PYTHONPATH=. python3 scripts/run_pipeline.py --mode live

fertility:
	@test -n "$$COHERE_API_KEY" || (echo "COHERE_API_KEY is not set. Get a free trial key: https://dashboard.cohere.com/api-keys" && exit 1)
	PYTHONPATH=. python3 scripts/measure_fertility.py

clean:
	rm -rf cache/responses cache/call_count.json cache/dry-run results/*.parquet figures/*.png
