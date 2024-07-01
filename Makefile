quality:
	ruff check
	ruff format --check

style:
	ruff check --fix
	ruff format

test-cov:
	pytest --cov-report term --cov-report xml:coverage.xml --cov=sentence_transformers