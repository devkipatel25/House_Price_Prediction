import os
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):
    # training the model for one time
    
    df = pd.read_csv("king_county_USA_house_data.csv")

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    test.drop("price", axis=1).to_csv("input.csv", index = False)
    test[['price']].to_csv('price.csv', index = False)
    housing = train.copy()

    # seprating the features and the labels
    housing_labels = housing["price"].copy()
    housing_features = housing.drop('price', axis=1)

    num_attributes = housing_features.columns.tolist()
    cat_attributes = []

    pipeline = build_pipeline(num_attributes, cat_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("model is trained")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data["price"] = predictions

    input_data.to_csv("output.csv", index = False)

    print("interface is completed, result is saved to output.csv")