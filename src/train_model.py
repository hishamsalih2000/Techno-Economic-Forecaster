# src/train_model.py

import pandas as pd
import numpy as np
import joblib
import pathlib
import sqlite3

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import randint

import config
from .utils import plot_results

# --- Main Training and Evaluation Function ---
def train_and_evaluate_model(df, features, target_column, model_save_path, unit_name):
    """
    Trains, evaluates, analyzes, and saves a regression model using RandomizedSearchCV.
    """
    print(f"\n{'='*20} Starting Workflow for Target: {target_column} {'='*20}")

    X = df[features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    categorical_features = ['Location_Name']
    numerical_features = ['Farm_Size_ha', 'Borehole_Depth_m', 'Total_Irr_Req_mm_per_ha']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', RandomForestRegressor(random_state=config.RANDOM_STATE))])

    param_distributions = {
        'model__n_estimators': randint(50, 201),
        'model__max_depth': [None] + list(range(10, 31, 5)),
        'model__min_samples_leaf': randint(1, 5)
    }
    
    random_search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=25, cv=5, scoring='r2', 
        n_jobs=-1, random_state=config.RANDOM_STATE
    )
    
    print("Starting RandomizedSearchCV...")
    random_search.fit(X_train, y_train)
    print("Search complete.")

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Evaluation Results ---")
    print(f"Best Hyperparameters: {random_search.best_params_}")
    print(f"Best Cross-Validated R²: {random_search.best_score_:.4f}")
    print("--- Final Evaluation on Unseen Test Set ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} {unit_name}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} {unit_name}")

    # --- Feature Importance Analysis ---
    try:
        num_features = best_model.named_steps['preprocessor'].transformers_[0][2]
        cat_features = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_features = list(num_features) + list(cat_features)
        importances = best_model.named_steps['model'].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\n--- Feature Importance ---")
        print(importance_df)
    except Exception as e:
        print(f"\nCould not calculate feature importance: {e}")

    # --- Save Model and Performance Plot ---
    joblib.dump(best_model, model_save_path)
    print(f"\nModel for {target_column} saved to: {model_save_path}")
    
    plot_results(
    y_true=y_test, 
    y_pred=y_pred, 
    target_column=target_column, 
    model_save_path=pathlib.Path(model_save_path), 
    r2_score=r2, 
    mae_score=mae, 
    unit_name=unit_name
    )
    
    print(f"{'='*60}\n")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Full Model Training and Evaluation Pipeline ---")
    
    # --- Read from SQL Database ---
    try:
        print(f"Connecting to database to load training data...")
        con = sqlite3.connect(config.DATABASE_PATH)
        # Write a simple SQL query to select all data from our table
        query = f"SELECT * FROM {config.TABLE_NAME}"
        # Use pandas' read_sql_query to load the data directly into a DataFrame
        main_df = pd.read_sql_query(query, con)
        con.close()
        print(f"Successfully loaded dataset with {len(main_df)} entries from the database.")
    except Exception as e:
        print(f"ERROR: Could not load data from the database. {e}")
        print("Please make sure you have run 'src/aggregate_results.py' first.")
        exit()

    # --- Train Model 1: Predict PV System Size ---
    train_and_evaluate_model(
        df=main_df,
        features=config.FEATURES,
        target_column='Required_PV_Size_kW',
        model_save_path=config.PV_MODEL_PATH,
        unit_name='kW'
    )

    # --- Train Model 2: Predict Net Present Cost (NPC) ---
    train_and_evaluate_model(
        df=main_df,
        features=config.FEATURES,
        target_column='NPC_USD',
        model_save_path=config.NPC_MODEL_PATH,
        unit_name='USD'
    )

    print("--- All models trained successfully. Script finished. ---")