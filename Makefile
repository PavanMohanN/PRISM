# PRISM — convenience targets
.PHONY: install dev smoke full assets test clean

install:           ## editable install
	pip install -e .

dev:               ## install with test extras
	pip install -e ".[dev]"

smoke:             ## fast CPU reproduction (experiments + assets)
	python reproduce.py --smoke

full:              ## paper-scale reproduction (GPU, multi-seed)
	python reproduce.py --full

assets:            ## rebuild tables+figures from cached results/
	python reproduce.py --assets-only

test:              ## run the test suite
	pytest -q

clean:             ## remove caches and generated artifacts
	rm -rf results/*.json results/*.npz paper_assets/*.pdf paper_assets/*.tex
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
