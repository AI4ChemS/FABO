# Self-Driving Discovery of Electric Vehicle Coolants

<p align="left">
  <img src="figures/image1.png" width="1500">
</p>


This repository contains the implementation of a self-driving laboratory (SDL) framework for the discovery and optimization of electric vehicle (EV) coolant mixtures.

The platform integrates:

- Automated search-space construction
- Machine learning-based surrogate modeling
- Bayesian Optimization (BO)
- Optional feature selection (FABO)
- Iterative experimental updating
- Mixture non-linearity handling

The objective is to autonomously propose optimal coolant mixtures using a closed-loop experimental workflow.

---

# Repository Structure

.
├── data/
│   ├── compounds.csv              # Pure component metadata (required)
│   └── train_data/                # Optional hot-start dataset
│
├── notebooks/
│   └── main.ipynb                 # Demo notebook (recommended entry point)
│
├── src/
│   ├── __init__.py
│   ├── BO.py                      # Core Bayesian Optimization engine
│   ├── feature_selection.py       # Feature selection utilities
│   ├── search_space_init.py       # Search space construction
│   └── sdl.py                     # Self-driving loop wrapper
│
├── requirements.txt
└── README.md

---

# Installation

1. Clone the repository:

git clone <your-repo-url>
cd <repo-folder>

2. Create environment (recommended: Conda):

conda create -n evcoolants python=3.10
conda activate evcoolants

3. Install dependencies:

pip install -r requirements.txt

---

# How to Run a Demo

The easiest way to run the framework is via:

notebooks/main.ipynb

Launch Jupyter from the repository root:

jupyter notebook

Then open:

notebooks/main.ipynb

Run cells sequentially.

---

# Workflow Overview

## 1. Search Space Initialization

cfg = SearchSpaceConfig(
    compounds_csv="../data/compounds.csv",
    processed_seed_csv="../data/train_data/train_data.csv",  # optional hot-start
    out_df_csv="../data/processed.csv",
    out_X_csv="../data/featurized_processed.csv",
    out_y_csv="../data/labels.csv",
    max_components=3,
    include_binary=True,
    include_ternary=True,
)

X, y, df = init_search_space(cfg)

This step:
- Generates mixture combinations
- Builds the featurized design matrix
- Creates the labels file
- Supports optional hot-start

---

## 2. Bayesian Optimization Loop

Configured using:

args = BOArgs(
    df_path=Path("../data/processed.csv"),
    compounds_path=Path("../data/compounds.csv"),
    df_featurized_path=Path("../data/featurized_processed.csv"),
    labels_path=Path("../data/labels.csv"),
    nb_iterations=200,
    FABO=True,
    FS_method="mRMR",
    min_features=5,
    max_features=10,
)

Run:

run_sdl_bo(args, CHEMICALS)

Each iteration:
1. Fits Gaussian Process surrogate
2. Optimizes acquisition function (EI / exploration)
3. Proposes new mixture
4. Receives experimental FOM
5. Updates dataset
6. Repeats

---

# Key Features

- Supports hot-start experiments
- Modular acquisition strategies
- Optional feature selection (FABO)
- Synergy penalty for linear mixtures
- Resumable optimization
- Designed for integration with hardware control

---

# Developers

Developed as part of the Self-Driving Discovery of Electric Vehicle Coolants project.

Mahyar Rajabi Kochi - AI4ChemS - University of Toronto

---

# Notes

- All data files (processed.csv, labels.csv, featurized_processed.csv) must remain row-aligned.
- data/compounds.csv is required to initialize the search space.
- Experimental control functions must be provided externally when deploying with real hardware.

For questions or collaboration inquiries, please open an issue or contact the developers.
