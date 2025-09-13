import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
import numpy as np

warnings.filterwarnings('ignore')

def train_demand_prediction_model():
    """
    Loads data, aggregates to route-level demand, engineers features, trains an XGBoost model with cross-validation,
    evaluates it, analyzes feature importance, and saves the final model.
    """
    print("--- 🤖 Starting Model Training Process ---")

    # 1. Load Data
    try:
        df = pd.read_csv("data/historical.csv")
        routes_df = pd.read_csv("data/routes.csv")
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: historical_ticket_sales.csv or routes.csv not found.")
        print("Please ensure data files are present.")
        return

    # 2. Data Cleaning & Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Filter incomplete hours (e.g., hour 6 has only 2 entries)
    hour_counts = df.groupby(df['timestamp'].dt.hour)['stop_id'].count()
    complete_hours = hour_counts[hour_counts == 10].index  # Expect 10 entries/hour (5 stops x 2 routes)
    if len(complete_hours) == 0:
        print("❌ Error: No complete hours in data. Cannot train model.")
        return
    df = df[df['timestamp'].dt.hour.isin(complete_hours)]
    
    df.dropna(subset=['passengers_boarded'], inplace=True)
    df['passengers_boarded'] = pd.to_numeric(df['passengers_boarded'], errors='coerce')
    df = df.dropna(subset=['passengers_boarded'])
    
    # Remove outliers
    upper_limit = df['passengers_boarded'].quantile(0.99)
    df = df[df['passengers_boarded'] <= upper_limit]
    
    # Aggregate to total passengers per route per timestamp
    df = df.groupby(['timestamp', 'route_id'])['passengers_boarded'].sum().reset_index()
    print(f"✅ Data cleaned and aggregated. New shape: {df.shape}")

    # 3. Feature Engineering
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Add lagged demand (previous hour's demand per route)
    df = df.sort_values(['route_id', 'timestamp'])
    df['prev_hour_demand'] = df.groupby('route_id')['passengers_boarded'].shift(1)
    df['prev_hour_demand'] = df['prev_hour_demand'].fillna(df['passengers_boarded'].mean())
    
    # Add route-specific feature: number of stops
    df['num_stops'] = df['route_id'].map(
        routes_df.set_index('route_id')['stops'].apply(lambda x: len(str(x).split(';')))
    )
    
    # Add interaction feature: hour_of_day * route_id
    df['route_hour_interaction'] = df['route_id'] * df['hour_of_day']
    
    print("✅ Features engineered: 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction'")

    # 4. Model Training
    features = ['route_id', 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction']
    target = 'passengers_boarded'
    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    print(f"⏳ Training XGBRegressor model with cross-validation...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )
    
    # Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_mae_scores = []
    cv_r2_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_cv_train, y_cv_train)
        y_cv_pred = model.predict(X_cv_val)
        cv_mae_scores.append(mean_absolute_error(y_cv_val, y_cv_pred))
        cv_r2_scores.append(r2_score(y_cv_val, y_cv_pred))
    
    model.fit(X_train, y_train)  # Final training on full training set
    print("✅ Model training complete.")

    # 5. Model Evaluation
    print("\n--- 📊 Model Evaluation ---")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE) on Test Set: {mae:.2f} passengers")
    print(f"R-squared (R²) Score on Test Set: {r2:.2f}")
    print(f"Cross-Validation MAE: {np.mean(cv_mae_scores):.2f} (±{np.std(cv_mae_scores):.2f})")
    print(f"Cross-Validation R²: {np.mean(cv_r2_scores):.2f} (±{np.std(cv_r2_scores):.2f})")
    
    # Feature Importance
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importance:")
    for feat, importance in feature_importance.items():
        print(f"  {feat}: {importance:.4f}")
    
    # Example Prediction
    sample_input = pd.DataFrame([[1, 8, 0, 50.0, 5, 1 * 8]], columns=features)  # Route 1, 8 AM, Monday, est. prev demand
    sample_prediction = model.predict(sample_input)[0]
    print(f"\nExample Prediction: For Route 1 at 8 AM on a Monday,")
    print(f"predicted total passengers are ~{sample_prediction:.0f}")

    # 6. Save the Model
    model_filename = "data/demand_predictor.pkl"
    joblib.dump(model, model_filename)
    print(f"\n--- 💾 Model Saved ---")
    print(f"✅ Trained model saved as '{model_filename}'")

if __name__ == "__main__":
    train_demand_prediction_model()