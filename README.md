# Battery Capacity Prediction

This project implements a machine learning solution for predicting battery capacity based on impedance measurements. The implementation includes data preprocessing, model training, evaluation, and capacity classification.

## Project Structure

```
Battery-Capacity/
├── src/
│   ├── data_loader.py      # Data loading and initial exploration
│   ├── preprocessor.py     # Data preprocessing and feature selection
│   ├── model_trainer.py    # Multiple regression models implementation
│   ├── evaluator.py        # Model evaluation and visualization
├── models/                 # Directory for saved models
├── results/               # Directory for plots and results
├── main.ipynb               # Main script to run the analysis
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Features

- Data preprocessing with standardization and feature selection
- Multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Support Vector Regression
- Comprehensive model evaluation
- Capacity classification into 5 bins
- Visualization of results

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset file (interview-dataset.csv.xlsx) in the project root directory
2. Run the analysis:
   ```bash
   jupyter notebook main.ipynb
   ```
3. Check the results in the `results` directory and saved models in the `models` directory

## Model Evaluation

The project evaluates models using multiple metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R2) Score
- Classification accuracy and F1 score for capacity bins

## Capacity Classification

Batteries are classified into 5 bins based on their capacity:
- Bin 1: ≤ 7000
- Bin 2: 7000 – 7400
- Bin 3: 7400 – 8000
- Bin 4: 8000 – 8500
- Bin 5: > 8500

## Output

The analysis generates:
- Model performance metrics
- Prediction vs. actual plots
- Residual analysis plots
- Classification accuracy and F1 Score reports
- Saved model files for future use
