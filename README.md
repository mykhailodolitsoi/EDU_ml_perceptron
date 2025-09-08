# Perceptron

This repository contains a simple Perceptron implementation for classifying the Iris dataset.  
It is a learning project based on concepts from the book [Machine Learning with PyTorch and Scikit-Learn](https://sebastianraschka.com/books/machine-learning-with-pytorch-and-scikit-learn/).

## Features
- Implements a basic Perceptron from scratch using Numpy.

- Trains on Iris dataset (Setosa and Versicolor).

- Visualizes decision regions and misclassification errors.

- Easy to extend to other binary classification tasks.

## Installation
First of all, you need to clone the perceptron:
```bash
git clone https://github.com/mykhailodolitsoi/PROJECT_ml_perceptron.git
```

Then create a venv(optionally) and install dependencies:
```bash
pip install -r requirements.txt
```

**You are welcome!**

## Usage
```python
from perceptron import Perceptron
import pandas as pd
import matplotlib.pyplot as plt

# Load Iris dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

# Prepare data
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)
X = df.iloc[0:100, [0, 2]].values

# Train Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# Visualize decision regions
plot_decision_regions(X, y, classifier=ppn)
plt.show()
```

## Result
- Shows scatter plot of two classes.
- Plots misclassification error over epochs.
- Displays decision regions.

## Description

### Class `Perceptron`

#### Parameters
- **`eta`** (`float`, default=0.01): Learning rate, controls the size of weight updates.  
- **`n_iter`** (`int`, default=50): Number of training epochs (full passes over the dataset).  
- **`random_state`** (`int`, optional): Random seed for reproducibility of weight initialization.  

---

#### Attributes (after fitting)
- **`w_`** (`ndarray`): Final weight vector of shape `(n_features,)`.  
- **`b_`** (`float`): Bias term.  
- **`errors_`** (`list` of `int`): Number of misclassifications per epoch.  

---

#### Methods
- **`__init__(self, eta=0.01, n_iter=50, random_state=1)`**  
  Constructor, initializes hyperparameters of the perceptron.

- **`fit(self, X, y)`**  
  Train the perceptron on dataset `X` (features) and `y` (labels).  
  - Initializes weights `w_` from a normal distribution with mean=0 and std=0.01.  
  - Iteratively updates weights and bias using the perceptron rule.  
  - Tracks the number of misclassifications in `errors_`.  
  - **Returns**: `self`.  

- **`net_input(self, X)`**  
  Computes the linear combination of inputs and weights:  
  \[
  z = X \cdot w + b
  \]  
  - **Returns**: scalar or array of net input values.  

- **`predict(self, X)`**  
  Applies the unit step function to `net_input`.  
  - **Returns**: Class label (`0` or `1`).  

### Helper Functions

#### `plot_decision_regions(X, y, classifier, resolution=0.02)`
Visualizes the decision boundaries of a trained classifier.  

- **Parameters**:  
  - `X`: Feature matrix (2D array).  
  - `y`: Target labels (1D array).  
  - `classifier`: Any object with a `.predict()` method (e.g., the Perceptron).  
  - `resolution`: Step size for creating the grid (default = 0.02).  

- **Description**:  
  1. Generates a mesh grid covering the feature space.  
  2. Predicts the class for every point on the grid using the given classifier.  
  3. Fills the background with colors representing different classes.  
  4. Overlays the original training samples with distinct markers.  

- **Output**:  
  No return value. Draws a `matplotlib` plot showing decision regions.  
