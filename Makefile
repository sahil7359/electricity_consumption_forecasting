.PHONY: install test lint fmt app backtest clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests app

fmt:
	ruff check --fix src tests app

app:
	streamlit run app/streamlit_app.py

backtest:
	elecforecast backtest

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info
