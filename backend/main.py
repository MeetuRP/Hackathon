import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timedelta
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# --- 🚌 App Initialization ---
app = FastAPI(
    title="Smart Bus Management System API",
    description="API for real-time bus tracking, scheduling, and on-demand model retraining.",
    version="1.2.0" # Version updated to reflect changes
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
    "route_data_with_stops": [],
    "static_schedule": {},
    "optimized_schedule": {}
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
    route_id: int
    trip_id: str
    latitude: float
    longitude: float
    speed_kmh: float
    occupancy: int
    timestamp: str

# --- 🤖 Integrated Model Training Logic ---
def train_demand_prediction_model():
    """
    Loads data, cleans it, engineers features, trains a model,
    evaluates it, saves the final model, and reloads it into the app state.
    """
    print("\n--- 🤖 Starting Background Model Training Process ---")
    try:
        df = pd.read_csv("data/historical_ticket_sales.csv")
    except FileNotFoundError:
        print("❌ Error: historical_ticket_sales.csv not found.")
        return

    df.dropna(subset=['passengers_boarded'], inplace=True)
    df = df[df['passengers_boarded'] != '']
    df['passengers_boarded'] = pd.to_numeric(df['passengers_boarded'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    upper_limit = df['passengers_boarded'].quantile(0.99)
    df = df[df['passengers_boarded'] <= upper_limit]
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    features = ['route_id', 'hour_of_day', 'day_of_week']
    target = 'passengers_boarded'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42, loss='squared_error'
    )
    model.fit(X_train, y_train)
    
    model_filename = "data/demand_predictor.pkl"
    joblib.dump(model, model_filename)
    
    app_state["prediction_model"] = model
    print("--- 🔄 Model training complete and reloaded into the live application. ---")


@app.on_event("startup")
def load_data_and_initialize_schedules():
    """
    This function includes the fix for the KeyError by re-inserting
    the stop_id into each stop's data dictionary.
    """
    print("--- 🚀 Server starting up! Loading data... ---")

    try:
        stops_df = pd.read_csv("data/stops.csv")
        routes_df = pd.read_csv("data/routes.csv")
        app_state["prediction_model"] = joblib.load("data/demand_predictor.pkl")
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Could not find data file: {e}")
        return
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load data: {e}")
        return

    # --- Defensive Data Processing ---
    required_stop_cols = ['stop_id', 'stop_name', 'latitude', 'longitude']
    if not all(col in stops_df.columns for col in required_stop_cols):
        print("❌ CRITICAL ERROR: stops.csv is missing required columns.")
        return
    stops_df = stops_df[required_stop_cols]

    stops_df.drop_duplicates(subset=['stop_id'], keep='first', inplace=True)
    stops_dict = stops_df.set_index('stop_id').to_dict('index')

    required_route_cols = ['route_id', 'route_name', 'stops']
    if not all(col in routes_df.columns for col in required_route_cols):
        print("❌ CRITICAL ERROR: routes.csv is missing required columns.")
        return
    app_state["routes_df"] = routes_df[required_route_cols]
    
    app_state["route_data_with_stops"] = []

    # Process each route with the fix applied
    for _, route_row in app_state["routes_df"].iterrows():
        stops_on_route_str = str(route_row.get('stops', ''))
        stop_ids = [int(s.strip()) for s in stops_on_route_str.split(';') if s.strip().isdigit()]

        # --- THIS IS THE FIX ---
        stops_list = []
        for stop_id in stop_ids:
            if stop_id in stops_dict:
                # Create a copy and add the stop_id back into the dictionary
                stop_details = stops_dict[stop_id].copy()
                stop_details['stop_id'] = stop_id
                stops_list.append(stop_details)
        # --- END OF FIX ---
        
        app_state["route_data_with_stops"].append({
            "route_id": int(route_row['route_id']),
            "route_name": str(route_row['route_name']),
            "stops": stops_list
        })
    
    generate_static_schedule()
    app_state["optimized_schedule"] = copy.deepcopy(app_state["static_schedule"])
    print("--- ✅ Data loaded and schedules initialized. Ready to go! ---")

def generate_static_schedule():
    """Generates a fixed timetable for all routes."""
    schedule = {}
    bus_counter = 1
    for route in app_state["route_data_with_stops"]:
        # CRITICAL FIX: Ensure route_id is valid before using as a key
        route_id = route.get('route_id')
        if route_id is None:
            continue # Skip routes with no ID

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
                    current_time += timedelta(minutes=5)
                
                schedule[route_id].append({
                    "trip_id": trip_id, "bus_id": f"BUS-{bus_counter:02d}", 
                    "depart_time": depart_time, "arrivals": arrivals
                })
                bus_counter += 1
    app_state["static_schedule"] = schedule

# --- 🚦 API Endpoints ---

@app.get("/api/routes", response_model=List[Route])
def get_routes():
    return app_state["route_data_with_stops"]

@app.get("/api/schedule/static")
def get_static_schedule():
    return app_state["static_schedule"]

@app.get("/api/schedule/optimized")
def get_optimized_schedule():
    return app_state["optimized_schedule"]

@app.get("/api/predictions/{route_id}")
def get_predictions(route_id: int):
    if route_id not in app_state["routes_df"]['route_id'].values:
        raise HTTPException(status_code=404, detail="Route not found")
    
    model = app_state["prediction_model"]
    now = datetime.now()
    predictions = []
    for i in range(4):
        future_time = now + timedelta(hours=i)
        hour, day_of_week = future_time.hour, future_time.weekday()
        prediction_input = pd.DataFrame([[route_id, hour, day_of_week]], columns=['route_id', 'hour_of_day', 'day_of_week'])
        predicted_passengers = model.predict(prediction_input)[0]
        predictions.append({"hour": hour, "predicted_passengers": round(predicted_passengers)})
        
    return {"route_id": route_id, "generated_at": now.isoformat(), "predictions": predictions}

@app.post("/api/live_update")
def live_update(update: BusUpdate):
    print(f"Received update for {update.bus_id} on trip {update.trip_id}")
    return {"status": "success", "message": f"Live update for {update.bus_id} received."}

@app.post("/api/retrain_model")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_demand_prediction_model)
    return {"message": "Model retraining started in the background. Check server logs for progress."}

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))  # Render sets $PORT dynamically
    uvicorn.run("main:app", host="0.0.0.0", port=port)
