# thesis_kedro_demo

A small Kedro project which takes a raw medical dataset and turns it into a trained model using the data catalog, two pipelines and a parameters file.

## What the project does

The project predicts whether a breast tumour is malignant or benign. It uses measurements of cell nuclei from the Breast Cancer Wisconsin (Diagnostic) dataset. The pipelines check and clean the raw data, split it into training and test sets, train a logistic regression model and save the evaluation results.

## Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **License:** CC BY 4.0, according to the UCI page

The data has 569 samples (357 benign, 212 malignant) and 30 numeric features. The original file has no header row, so the column names are added in the first node. The file is small, so it is included in the repository as `data/01_raw/wdbc.csv`.

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/PRO-D-PJATK/Kedro_s33277.git
cd Kedro_s33277
python -m venv .venv
```

Activate the environment:

- Git Bash on Windows: `source .venv/Scripts/activate`
- PowerShell on Windows: `.venv\Scripts\Activate.ps1`
- cmd on Windows: `.venv\Scripts\activate.bat`
- Linux / macOS: `source .venv/bin/activate`

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## How to run

```bash
kedro run
kedro run --pipelines=data_processing
kedro run --pipelines=model
```

The first command runs everything. The `model` pipeline needs the output of `data_processing`, so run that one first or run the whole project.

## Pipelines

**data_processing**
1. `validate_raw_data` checks the shape, missing values and labels and adds the column names.
2. `clean_data` removes the ID column and duplicates and encodes the diagnosis (malignant = 1, benign = 0).
3. `split_data` creates a stratified train/test split.

**model**
4. `train_model` scales the features and trains a logistic regression.
5. `evaluate_model` calculates the metrics and saves them to `data/08_reporting/metrics.json`.

## Configuration

All datasets are defined in `conf/base/catalog.yml`, so the node code contains no file paths. The tunable values are in `conf/base/parameters.yml`:

- `random_state` (seed, default 42)
- `test_size` (default 0.2)
- `model_options.C` and `model_options.max_iter` (logistic regression settings)
- `data_validation` (expected shape of the raw file)

## Results

With the default parameters the model reaches an accuracy of 0.965 on 114 test samples. Precision is 0.975, recall is 0.929 and F1 is 0.951. Three malignant tumours were missed and one benign tumour was classified as malignant. The test set is small, so these numbers can change a bit with another split.

## Reproducibility

- Python version: 3.11.3
- Kedro version: 1.7.0
- The exact package versions are pinned in `requirements.txt`. To recreate the environment, create a virtual environment and run `pip install -r requirements.txt`.
- The random seed is stored in `parameters.yml`. Running `kedro run` twice gives the same `metrics.json`.
- No secrets are used in this project. Local settings would go to `conf/local/`, which Git ignores.
