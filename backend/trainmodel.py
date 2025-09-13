import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_demand_prediction_model():
    """
    Loads data, cleans it, engineers features, trains a model,
    evaluates it, and saves the final model.
    """
    print("--- 🤖 Starting Model Training Process ---")

    # 1. Load Data
    try:
        df = pd.read_csv("data\historical_ticket_sales.csv")
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: historical_ticket_sales.csv not found.")
        print("Please run the data generation script first.")
        return

    # 2. Data Cleaning & Preprocessing
    # Drop rows where passenger count is missing
    df.dropna(subset=['passengers_boarded'], inplace=True)
    df = df[df['passengers_boarded'] != '']
    
    # Convert data types
    df['passengers_boarded'] = pd.to_numeric(df['passengers_boarded'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Remove outliers (using a simple percentile-based approach)
    upper_limit = df['passengers_boarded'].quantile(0.99)
    df = df[df['passengers_boarded'] <= upper_limit]
    print(f"✅ Data cleaned. New shape: {df.shape}")

    # 3. Feature Engineering
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek # Monday=0, Sunday=6
    
    print("✅ Features engineered: 'hour_of_day', 'day_of_week'")

    # 4. Model Training
    # Define features (X) and target (y)
    features = ['route_id', 'hour_of_day', 'day_of_week']
    target = 'passengers_boarded'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"⏳ Training GradientBoostingRegressor model...")
    # Initialize and train the model
    # Using fewer estimators for faster training in a hackathon context
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        loss='squared_error'
    )
    
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # 5. Model Evaluation
    print("\n--- 📊 Model Evaluation ---")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae:.2f} passengers")
    print(f"R-squared (R²) Score: {r2:.2f}")
    
    # Example Prediction
    sample_input = pd.DataFrame([[101, 8, 0]], columns=features) # Route 101, 8 AM on a Monday
    sample_prediction = model.predict(sample_input)[0]
    print(f"\nExample Prediction: For Route 101 at 8 AM on a Monday,")
    print(f"predicted passengers are ~{sample_prediction:.0f}")

    # 6. Save the Model
    model_filename = "demand_predictor.pkl"
    joblib.dump(model, model_filename)
    print(f"\n--- 💾 Model Saved ---")
    print(f"✅ Trained model saved as '{model_filename}'")


if __name__ == "__main__":
    train_demand_prediction_model()