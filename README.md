# Diabetes Disease Progression Prediction

This repository contains a Jupyter Notebook (`Toy_dataset_02.ipynb`) that demonstrates a basic Linear Regression model to predict diabetes disease progression using the scikit-learn diabetes dataset.

## Table of Contents

* [Overview](#overview)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Running the Notebook](#running-the-notebook)
* [Notebook Contents](#notebook-contents)
* [Results](#results)
* [Libraries Used](#libraries-used)

---

## Overview

This project aims to predict the progression of diabetes in patients based on various features. It uses the `load_diabetes` dataset available in scikit-learn, which consists of 10 baseline variables (age, sex, body mass index, average blood pressure, and six blood serum measurements) for 442 diabetes patients, along with a quantitative measure of disease progression one year after baseline.

The notebook covers the following steps:
1.  **Loading the Dataset:** Imports the `diabetes` dataset.
2.  **Splitting Data:** Divides the dataset into training (80%) and testing (20%) sets.
3.  **Model Training:** Trains a Linear Regression model on the training data.
4.  **Model Evaluation:** Evaluates the model's performance using Mean Squared Error (MSE) and R-squared ($R^2$) metrics.
5.  **Prediction on New Data:** Demonstrates how to make predictions for a new patient.
6.  **Visualization:** Presents a scatter plot comparing actual vs. predicted disease progression.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need Python installed on your system. It's recommended to use Anaconda or Miniconda for environment management.

* Python 3.x
* Jupyter Notebook or Jupyter Lab

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Suchendra13/Toy_dataset_02.git](https://github.com/Suchendra13/Toy_dataset_02.git)
    ```
2.  **Navigate into the project directory:**
    ```bash
    cd Toy_dataset_02
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
4.  **Install the required libraries:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn ipykernel
    ```
    *Note: The notebook mentions `conda env:base` as the kernel. If you're using Anaconda, you might already have these, or you can activate your base environment: `conda activate base`.*

### Running the Notebook

1.  **Start Jupyter Notebook or Jupyter Lab:**
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
2.  Your web browser will open a new tab with the Jupyter interface.
3.  Click on `Toy_dataset_02.ipynb` to open and run the notebook.

## Notebook Contents

* `Toy_dataset_02.ipynb`: The main Jupyter Notebook file containing the data loading, model training, evaluation, prediction, and visualization code.

## Results

The notebook provides evaluation metrics for the Linear Regression model:

* **Mean Squared Error (MSE):** A measure of the average of the squares of the errors. Lower values mean better fit.
* **R-squared ($R^2$):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A value closer to 1.0 indicates a better fit.

--- Model Evaluation ---
*Mean Squared Error: 3069.81
*R-squared (R²): 0.40

--- Example Prediction ---
Data for new patient: [ 0.03807591  0.05068012  0.06169621  0.02187239 -0.0442235  -0.03482076
-0.04340085 -0.00259226  0.01990749 -0.01764613]
Predicted disease progression: 209.8

## Libraries Used

The following Python libraries are used in this project:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn` (specifically `load_diabetes`, `train_test_split`, `LinearRegression`, `mean_squared_error`, `r2_score`)
