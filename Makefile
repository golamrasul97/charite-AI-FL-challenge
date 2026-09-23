PYTHON ?= python

.PHONY: install run evaluate test clean

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	mkdir -p outputs/artifacts
	$(PYTHON) run_submission.py --train data/train.jsonl --input data/validation_inputs.jsonl --output outputs/validation_predictions.jsonl --artifacts-dir outputs/artifacts

evaluate: run
	$(PYTHON) evaluator/evaluate.py --inputs data/validation_inputs.jsonl --ground-truth data/validation_ground_truth.jsonl --predictions outputs/validation_predictions.jsonl --report outputs/validation_report.json

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf outputs .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
