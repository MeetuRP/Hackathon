import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any
from datetime import datetime, timedelta
import copy
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import haversine as hs
import os
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
import bcrypt

load_dotenv()  # Load environment variables from .env file
url: str = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
try:
    supabase: Client = create_client(url, key)
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to initialize Supabase client: {e}")
    supabase = None

warnings.filterwarnings('ignore')

# --- 🚌 App Initialization ---
app = FastAPI(
    title="Smart Bus Management System API",
    description="API for real-time bus tracking, scheduling, and on-demand model retraining.",
    version="1.5.4"  # Robust timestamp handling
)

# --- 🌐 CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 💾 In-Memory Data Storage ---
app_state: Dict[str, Any] = {
    "routes_df": None,
    "stops_df": None,
    "prediction_model": None,
    "bus_prediction_model": None,  # Added for the new model
    "route_data_with_stops": [],
    "static_schedule": {},
    "optimized_schedule": {},
    "live_bus_locations": {}
}

# --- 📦 Pydantic Models ---
class Stop(BaseModel):
    stop_id: int
    stop_name: str
    latitude: float
    longitude: float

class Route(BaseModel):
    route_id: int
    route_name: str
    stops: List[Stop]

class BusUpdate(BaseModel):
    bus_id: str
    route_id: str
    trip_id: str
    latitude: float
    longitude: float
    speed_kmh: float
    occupancy: int
    timestamp: str

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class BusPredictionRequest(BaseModel):
    route_id: int
    hour_of_day: int
    day_of_week: int
    prev_hour_demand: int
    num_stops: int
    route_hour_interaction: int

# --- 🤖 Integrated Model Training Logic ---
def train_demand_prediction_model():
    """
    Loads data, aggregates to route-level demand, engineers features, trains an XGBoost model with cross-validation,
    evaluates it, analyzes feature importance, and saves the final model.
    This function is designed to be called by the API for on-demand retraining.
    """
    print("--- 🤖 Starting Model Training Process ---")

    # 1. Load Data
    try:
        df = pd.read_csv("data/historical.csv")
        routes_df = pd.read_csv("data/routes.csv")
        stops_df = pd.read_csv("data/stops.csv")
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: Could not find all required synthetic data files.")
        print("Please ensure all data files are present in 'data/'.")
        return

    # --- Robust Timestamp Handling ---
    # Find the timestamp column dynamically
    timestamp_col = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)
    if not timestamp_col:
        print("❌ CRITICAL ERROR: Timestamp column not found in historical data.")
        return
    
    print(f"Found timestamp column: '{timestamp_col}'")
    df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
    # --- End of Fix ---

    # 2. Data Cleaning & Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Merge with stops to get route_id for each ticket sale
    df = pd.merge(df, stops_df[['stop_id']], on='stop_id', how='left')
    df.dropna(subset=['route_id'], inplace=True) # Ensure all sales are associated with a route
    
    df.dropna(subset=['passengers_boarded'], inplace=True)
    df['passengers_boarded'] = pd.to_numeric(df['passengers_boarded'], errors='coerce')
    df.dropna(subset=['passengers_boarded'], inplace=True)
    
    # Remove outliers
    upper_limit = df['passengers_boarded'].quantile(0.99)
    df = df[df['passengers_boarded'] <= upper_limit]
    
    # Aggregate demand by timestamp and route
    df = df.groupby(['timestamp', 'route_id'])['passengers_boarded'].sum().reset_index()
    print(f"✅ Data cleaned and aggregated. New shape: {df.shape}")

    # 3. Feature Engineering
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    df = df.sort_values(['route_id', 'timestamp'])
    df['prev_hour_demand'] = df.groupby('route_id')['passengers_boarded'].shift(1)
    df['prev_hour_demand'] = df['prev_hour_demand'].fillna(df['passengers_boarded'].mean())
    
    # Calculate number of stops per route from routes_df
    routes_df['num_stops'] = routes_df['stops'].apply(lambda x: len(x.split(';')))
    df = pd.merge(df, routes_df[['route_id', 'num_stops']], on='route_id', how='left')
    df['num_stops'].fillna(0, inplace=True)

    df['route_hour_interaction'] = df['route_id'] * df['hour_of_day']
    
    print("✅ Features engineered: 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction'")

    # 4. Model Training
    features = ['route_id', 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction']
    target = 'passengers_boarded'
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"⏳ Training XGBRegressor model with cross-validation...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )
    
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
    
    model.fit(X_train, y_train)
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
    
    # 6. Save the Model
    model_filename = "data/demand_predictor.pkl"
    joblib.dump(model, model_filename)
    print(f"\n--- 💾 Model Saved ---")
    print(f"✅ Trained model saved as '{model_filename}'")
    # Reload the model into app_state after training
    app_state["prediction_model"] = model
    print("✅ Model reloaded into application state.")

def train_bus_prediction_model():
    """
    Loads historical data, engineers features, and trains an XGBoost model
    to predict the number of buses required based on passenger demand.
    """
    print("--- 🚌 Starting Bus Prediction Model Training ---")
    BUS_CAPACITY = 50
    # 1. Load Data
    try:
        df = pd.read_csv("data/historical.csv")
        routes_df = pd.read_csv("data/routes.csv")
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: Could not find all required data files.")
        return
    
    # Aggregate demand by timestamp and route
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_agg = df.groupby(['timestamp', 'route_id'])['passengers_boarded'].sum().reset_index()
    df_agg['number_of_buses'] = (df_agg['passengers_boarded'] / BUS_CAPACITY).apply(lambda x: int(x) + (1 if x % 1 != 0 else 0))


    # 2. Feature Engineering
    df_agg['hour_of_day'] = df_agg['timestamp'].dt.hour
    df_agg['day_of_week'] = df_agg['timestamp'].dt.dayofweek
    df_agg = df_agg.sort_values(['route_id', 'timestamp'])
    df_agg['prev_hour_demand'] = df_agg.groupby('route_id')['passengers_boarded'].shift(1).fillna(0)

    # Get number of stops for each route from routes.csv
    routes_df['num_stops'] = routes_df['stops'].apply(lambda x: len(x.split(';')))
    df_agg = pd.merge(df_agg, routes_df[['route_id', 'num_stops']], on='route_id', how='left')

    df_agg['route_hour_interaction'] = df_agg['route_id'] * df_agg['hour_of_day']
    print("✅ Features engineered.")

    # 3. Model Training
    features = ['route_id', 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction']
    target = 'number_of_buses'
    X = df_agg[features]
    y = df_agg[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"⏳ Training XGBRegressor model for bus prediction...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # 4. Model Evaluation
    print("\n--- 📊 Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_pred_rounded = [round(p) for p in y_pred]

    mae = mean_absolute_error(y_test, y_pred_rounded)
    r2 = r2_score(y_test, y_pred_rounded)

    print(f"Mean Absolute Error (MAE): {mae:.2f} buses")
    print(f"R-squared (R²) Score: {r2:.2f}")

    # 5. Save the Model
    model_filename = "data/bus_predictor.pkl"
    joblib.dump(model, model_filename)
    print(f"\n--- 💾 Model Saved ---")
    print(f"✅ Trained model saved as '{model_filename}'")
    # Reload the model into app_state after training
    app_state["bus_prediction_model"] = model
    print("✅ Bus prediction model reloaded into application state.")

@app.on_event("startup")
def load_data_and_initialize_schedules():
    print("--- 🚀 Server starting up! Loading data... ---")
    try:
        # Load data from the 'data' directory
        stops_df = pd.read_csv("data/stops.csv")
        routes_df = pd.read_csv("data/routes.csv")
        app_state["prediction_model"] = joblib.load("data/demand_predictor.pkl")
        app_state["bus_prediction_model"] = joblib.load("data/bus_predictor.pkl")
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Could not find data file: {e}")
        return
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load data: {e}")
        return

    # --- Data Processing Fix ---
    # Create a dictionary of stops for easy lookup
    stops_dict = stops_df.set_index('stop_id').to_dict('index')

    # Group stops by route_id
    routes_df['stops'] = routes_df['stops'].astype(str)
    stop_ids = routes_df['stops'].str.split(';').explode()
    stop_ids = pd.to_numeric(stop_ids)
    stops_by_route = stop_ids.groupby(routes_df['route_id']).apply(list).reset_index(name='stops')

    # Merge the stops list into the routes dataframe
    routes_with_stops_df = pd.merge(routes_df, stops_by_route, on='route_id', how='left')
    routes_with_stops_df['stops_y'] = routes_with_stops_df['stops_y'].fillna('').apply(list)
    routes_with_stops_df.rename(columns={'stops_y': 'stops_list'}, inplace=True)
    
    app_state["routes_df"] = routes_with_stops_df

    # Build the detailed route data with stop information
    app_state["route_data_with_stops"] = []
    for _, route_row in routes_with_stops_df.iterrows():
        stop_ids = route_row.get('stops_list', [])
        stops_list = []
        for stop_id in stop_ids:
            if stop_id in stops_dict:
                stop_details = stops_dict[stop_id].copy()
                stop_details['stop_id'] = stop_id
                # Ensure correct data types for Pydantic model
                stop_details['latitude'] = float(stop_details['latitude'])
                stop_details['longitude'] = float(stop_details['longitude'])
                stops_list.append(stop_details)
        
        app_state["route_data_with_stops"].append({
            "route_id": route_row['route_id'],
            "route_name": route_row['route_name'],
            "stops": stops_list
        })
    
    # Generate schedules with the corrected data
    generate_static_schedule()
    generate_optimized_schedule()
    print("--- ✅ Data loaded and schedules initialized. Ready to go! ---")


def generate_static_schedule():
    """Generates a fixed timetable for all routes."""
    schedule = {}
    bus_counter = 1
    for route in app_state["route_data_with_stops"]:
        route_id = route.get('route_id')
        if route_id is None:
            continue
        schedule[route_id] = []
        for hour in range(6, 22):
            trip_times = [0, 15, 30, 45] if (7 <= hour <= 10 or 17 <= hour <= 19) else [0, 30]
            for minute in trip_times:
                depart_time = f"{hour:02d}:{minute:02d}"
                trip_id = f"{route_id}-{hour:02d}{minute:02d}"
                arrivals = []
                current_time = datetime.strptime(depart_time, "%H:%M")
                for stop in route['stops']:
                    arrivals.append({"stop_id": stop['stop_id'], "arrival_time": current_time.strftime("%H:%M")})
                    current_time += timedelta(minutes=5) # Assuming 5 mins between stops
                schedule[route_id].append({
                    "trip_id": trip_id, "bus_id": f"BUS-{bus_counter:02d}",
                    "depart_time": depart_time, "arrivals": arrivals
                })
        bus_counter += 1
    app_state["static_schedule"] = schedule

def generate_optimized_schedule():
    """Generates an optimized timetable based on predicted demand."""
    if app_state["bus_prediction_model"] is None:
        print("❌ Model not loaded. Cannot generate optimized schedule.")
        return

    model = app_state["bus_prediction_model"]
    now = datetime.now()
    day_of_week = now.weekday()
    routes_df = app_state["routes_df"]
    schedule = {}
    bus_counter = 101 # Start from a different bus counter to distinguish
    for route in app_state["route_data_with_stops"]:
        route_id = route.get('route_id')
        if route_id is None:
            continue
        schedule[route_id] = []
        num_stops = len(route['stops'])
        # Get recent demand for lagged feature
        try:
            df = pd.read_csv("data/historical.csv")
            # --- Robust Timestamp Handling ---
            timestamp_col = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)
            if timestamp_col:
                df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                recent = df[df['route_id'] == route_id].groupby(df['timestamp'].dt.hour)['passengers_boarded'].sum().tail(1)
                prev_demand = recent.values[0] if not recent.empty else 50.0
            else:
                prev_demand = 50.0
        except Exception:
            prev_demand = 50.0
        for hour in range(6, 22):
            route_hour_interaction = route_id * hour
            prediction_input = pd.DataFrame(
                [[route_id, hour, day_of_week, prev_demand, num_stops, route_hour_interaction]],
                columns=['route_id', 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction']
            )
            predicted_buses = round(model.predict(prediction_input)[0])
            
            if predicted_buses >= 2:
                trip_minutes = [0, 15, 30, 45]  # High demand, more buses
            else:
                trip_minutes = [0, 30]  # Low demand, fewer buses
            
            for minute in trip_minutes:
                depart_time = f"{hour:02d}:{minute:02d}"
                trip_id = f"OPT-{route_id}-{hour:02d}{minute:02d}"
                arrivals = []
                current_time = datetime.strptime(depart_time, "%H:%M")
                for stop in route['stops']:
                    arrivals.append({"stop_id": stop['stop_id'], "arrival_time": current_time.strftime("%H:%M")})
                    current_time += timedelta(minutes=5)
                schedule[route_id].append({
                    "trip_id": trip_id, "bus_id": f"BUS-{bus_counter:03d}",
                    "depart_time": depart_time, "arrivals": arrivals
                })
        bus_counter += 1
    app_state["optimized_schedule"] = schedule
    print("--- 📅 Optimized schedule generated based on demand predictions. ---")

@app.get("/api/routes", response_model=List[Route])
def get_routes():
    return app_state["route_data_with_stops"]

@app.get("/api/schedule/static")
def get_static_schedule():
    return app_state["static_schedule"]

@app.get("/api/schedule/optimized")
def get_optimized_schedule():
    return app_state["optimized_schedule"]

@app.post("/api/retrain_model")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_demand_prediction_model)
    return {"message": "Model retraining started in the background. Check server logs for progress."}

@app.post("/api/retrain_bus_model")
def retrain_bus_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_bus_prediction_model)
    return {"message": "Bus prediction model retraining started in the background. Check server logs for progress."}

@app.post("/api/predict_buses")
def predict_buses(req: BusPredictionRequest):
    if app_state["bus_prediction_model"] is None:
        raise HTTPException(status_code=503, detail="Bus prediction model is not available.")
    
    model = app_state["bus_prediction_model"]
    
    prediction_input = pd.DataFrame(
        [[req.route_id, req.hour_of_day, req.day_of_week, req.prev_hour_demand, req.num_stops, req.route_hour_interaction]],
        columns=['route_id', 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction']
    )
    
    predicted_buses = round(model.predict(prediction_input)[0])
    
    return {"predicted_buses": predicted_buses}


# --- Authentication Endpoints ---
@app.post("/api/register")
def register(req: RegisterRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
    existing = supabase.table("users").select("*").eq("email", req.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user = {"username": req.username, "email": req.email, "password": hashed_pw}
    try:
        supabase.table("users").insert(user).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register user: {e}")
    return {"success": True, "message": "User registered successfully"}

@app.post("/api/login")
def login(req: LoginRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
    res = supabase.table("users").select("*").eq("email", req.email).execute()
    if not res.data:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user = res.data[0]
    stored_pw = user["password"]
    if not bcrypt.checkpw(req.password.encode("utf-8"), stored_pw.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"success": True, "message": "Login successful", "user": {"id": user["id"], "username": user["username"], "email": user["email"]}}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)