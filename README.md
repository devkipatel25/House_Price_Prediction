# House Price Prediction ML Pipeline

An end-to-end Machine Learning pipeline built with Python and Scikit-Learn to predict house prices using the **King County House Sales Dataset**. This repository demonstrates modular data preprocessing, cross-validation model evaluation, model serialization, and automated inference.

## 📌 Project Overview

Instead of running a basic Jupyter notebook, this project follows a modular, production-style architecture:
- **Data Preprocessing:** Handled via a Scikit-Learn `ColumnTransformer` pipeline including missing value imputation (`SimpleImputer`) and numerical scaling (`StandardScaler`).
- **Modeling:** Trained using a `RandomForestRegressor`.
- **Validation:** Evaluated using 10-fold Cross Validation (`cross_val_score`) with Root Mean Squared Error (RMSE).
- **Inference Workflow:** Pipelines and model weights are serialized into `.pkl` files using `joblib` to decouple training from model deployment.

---

## 📁 Repository Structure

```text
├── main.py                          # Complete pipeline: training, saving models, & inference logic
├── king_county_USA_house_data.csv   # Primary training dataset
├── input.csv                        # Unseen feature dataset for inference
├── output.csv                       # Generated predictions file
├── price.csv                        # Ground truth actual prices for test evaluation
├── pipeline.pkl                     # Serialized Scikit-Learn preprocessing pipeline
└── README.md                        # Project documentation
