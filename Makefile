.PHONY: install test lint demo clean
install:
	python -m pip install -e '.[dev]'
test:
	pytest -q
lint:
	ruff check src tests
demo:
	python -m nufield.cli demo --output-dir outputs/demo --n-sims 160 --grid-size 12 --seed 7
clean:
	rm -rf outputs .pytest_cache .ruff_cache build dist *.egg-info
