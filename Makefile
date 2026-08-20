.PHONY: install ingest eval docker-up docker-down test lint clean

install:
	pip install -r backend/requirements.txt

ingest:
	python scripts/fetch_hf.py --sample 500 --out data/raw/jobs.jsonl
	python scripts/build_qrels.py --jobs data/raw/jobs.jsonl --cv data/sample/cv_sample.txt --out data/qrels.jsonl
	python scripts/ingest.py --jobs data/raw/jobs.jsonl

eval:
	python backend/app/evaluation/compare.py --jobs data/raw/jobs.jsonl --qrels data/qrels.jsonl --out artifacts/metrics.json
	cat artifacts/metrics.json

eval-full:
	python backend/app/evaluation/compare.py --jobs data/raw/jobs.jsonl --qrels data/qrels.jsonl --with-ce --out artifacts/metrics-full.json

docker-up:
	docker compose -f infra/docker-compose.yml up --build

docker-down:
	docker compose -f infra/docker-compose.yml down -v

test:
	pytest backend/tests -q

lint:
	ruff check backend/ scripts/
	ruff format --check backend/ scripts/

clean:
	rm -rf artifacts/ .cache/ __pycache__/
