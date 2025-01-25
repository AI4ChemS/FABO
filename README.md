
# FABO: Feature-Adaptive Bayesian Optimization Framework

![License](https://img.shields.io/github/license/AI4ChemS/FABO) 
![Issues](https://img.shields.io/github/issues/AI4ChemS/FABO) 
![Stars](https://img.shields.io/github/stars/AI4ChemS/FABO)

## Overview

**FABO (Feature-Adaptive Bayesian Optimization)** is an advanced framework designed to enhance material discovery by dynamically selecting the most relevant features at each iteration of the optimization process. This novel integration of feature selection techniques within the **Bayesian Optimization (BO)** loop ensures reduced data dimensionality and increased efficiency, especially in high-dimensional search spaces.

![Project Demo](figures/1.svg)

This repository provides the code and resources to run FABO on tasks such as **CO₂ uptake** and **electronic property optimization** for **MOF discovery**. FABO adapts to data distributions, making it more flexible than traditional BO methods reliant on static feature sets.

---

## Key Features

- **Dynamic Feature Selection**: Updates feature sets iteratively to improve optimization efficiency.
- **Flexible Acquisition Strategies**: Supports **Expected Improvement (EI)** and **Upper Confidence Bound (UCB)**, enabling better exploration and exploitation.
- **Robust Surrogate Model**: Employs a **Gaussian Process** to predict outcomes and quantify uncertainty.
- **Task-Agnostic**: Applicable to diverse optimization problems, including adsorption and electronic property optimization.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/AI4ChemS/FABO.git
cd FABO
pip install -r requirements.txt
```

---

## Customizing FABO for Your Application

FABO is highly customizable. In the **FABO_main** notebook, you can adjust parameters to align with your specific optimization needs:

- **`nb_iterations`**: Number of Bayesian optimization iterations.
- **`nb_initialization`**: Number of random initial samples for optimization.
- **`min_features` and `max_features`**: Range of features selected per iteration (default: `5` to `20`).
- **`FABO`**: Boolean flag to enable (`True`) or disable (`False`) Feature-Adaptive Bayesian Optimization.
- **`which_acquisition`**: Acquisition function choice: `"EI"`, `"max y_hat"`, or `"max sigma"`.
- **`FS_method`**: Feature selection method (`'spearman'` or `'mRMR'`).
- **`n_seed` and `seeds`**: Control randomness in optimization (default: `19` seeds).
- **`path`**: Path to the dataset (CSV format).
- **`label`**: Target property for optimization (e.g., `"pure_uptake_CO2_298.00_1600000"`).

### Example Usage

To optimize a new task, modify these parameters:
1. **`path`**: Set to your dataset file path (e.g., `"data/my_dataset.csv"`).
2. **`label`**: Define the property to optimize (e.g., `"target_property"`).
3. **`which_acquisition`**: Choose based on your preference for exploration (`"max sigma"`) or exploitation (`"max y_hat"`).
4. **`min_features`** and **`max_features`**: Adjust to your task's feature range.

By tailoring these settings, FABO can adapt to efficiently optimize material discovery tasks.

---

Feel free to reach out if you have questions or need assistance!
