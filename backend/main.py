import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timedelta
import copy
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
import haversine as hs

warnings.filterwarnings('ignore')

# --- 🚌 App Initialization ---
app = FastAPI(
    title="Smart Bus Management System API",
    description="API for real-time bus tracking, scheduling, and on-demand model retraining.",
    version="1.4.1"  # Updated to reflect feature alignment fix
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
    Loads data, aggregates to route-level demand, engineers features, trains an XGBoost model,
    saves it, and reloads it into the app state.
    """
    print("\n--- 🤖 Starting Background Model Training Process ---")
    try:
        df = pd.read_csv("data/historical_ticket_sales.csv")
        routes_df = pd.read_csv("data/routes.csv")
    except FileNotFoundError:
        print("❌ Error: historical_ticket_sales.csv or routes.csv not found.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Drop incomplete hours (e.g., hour 6)
    hour_counts = df.groupby(df['timestamp'].dt.hour)['stop_id'].count()
    complete_hours = hour_counts[hour_counts == 10].index
    df = df[df['timestamp'].dt.hour.isin(complete_hours)]

    df.dropna(subset=['passengers_boarded'], inplace=True)
    df['passengers_boarded'] = pd.to_numeric(df['passengers_boarded'])
    upper_limit = df['passengers_boarded'].quantile(0.99)
    df = df[df['passengers_boarded'] <= upper_limit]

    # Aggregate to total passengers per route per timestamp
    df = df.groupby(['timestamp', 'route_id'])['passengers_boarded'].sum().reset_index()
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Add lagged demand
    df = df.sort_values(['route_id', 'timestamp'])
    df['prev_hour_demand'] = df.groupby('route_id')['passengers_boarded'].shift(1)
    df['prev_hour_demand'] = df['prev_hour_demand'].fillna(df['passengers_boarded'].mean())

    # Add number of stops
    df['num_stops'] = df['route_id'].map(
        routes_df.set_index('route_id')['stops'].apply(lambda x: len(str(x).split(';')))
    )

    # Add interaction feature
    df['route_hour_interaction'] = df['route_id'] * df['hour_of_day']

    features = ['route_id', 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction']
    target = 'passengers_boarded'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)

    model_filename = "data/demand_predictor.pkl"
    joblib.dump(model, model_filename)

    app_state["prediction_model"] = model
    generate_optimized_schedule()
    print("--- 🔄 XGBoost model trained, saved, and schedule optimized. ---")

@app.on_event("startup")
def load_data_and_initialize_schedules():
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
    for _, route_row in app_state["routes_df"].iterrows():
        stops_on_route_str = str(route_row.get('stops', ''))
        stop_ids = [int(s.strip()) for s in stops_on_route_str.split(';') if s.strip().isdigit()]
        stops_list = []
        for stop_id in stop_ids:
            if stop_id in stops_dict:
                stop_details = stops_dict[stop_id].copy()
                stop_details['stop_id'] = stop_id
                stops_list.append(stop_details)
        app_state["route_data_with_stops"].append({
            "route_id": int(route_row['route_id']),
            "route_name": str(route_row['route_name']),
            "stops": stops_list
        })

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
                    current_time += timedelta(minutes=5)
                schedule[route_id].append({
                    "trip_id": trip_id, "bus_id": f"BUS-{bus_counter:02d}",
                    "depart_time": depart_time, "arrivals": arrivals
                })
                bus_counter += 1
    app_state["static_schedule"] = schedule

def generate_optimized_schedule():
    """Generates an optimized timetable based on predicted demand."""
    if app_state["prediction_model"] is None:
        print("❌ Model not loaded. Cannot generate optimized schedule.")
        return

    model = app_state["prediction_model"]
    now = datetime.now()
    day_of_week = now.weekday()
    routes_df = app_state["routes_df"]
    schedule = {}
    bus_counter = 1
    for route in app_state["route_data_with_stops"]:
        route_id = route.get('route_id')
        if route_id is None:
            continue
        schedule[route_id] = []
        num_stops = len(route['stops'])
        # Get recent demand for lagged feature
        try:
            df = pd.read_csv("data/historical_ticket_sales.csv")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            recent = df[df['route_id'] == route_id].groupby(df['timestamp'].dt.hour)['passengers_boarded'].sum().tail(1)
            prev_demand = recent.values[0] if not recent.empty else 50.0
        except:
            prev_demand = 50.0  # Default if no data
        for hour in range(6, 22):
            route_hour_interaction = route_id * hour
            prediction_input = pd.DataFrame(
                [[route_id, hour, day_of_week, prev_demand, num_stops, route_hour_interaction]],
                columns=['route_id', 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction']
            )
            predicted = model.predict(prediction_input)[0]
            if predicted > 70:
                trip_minutes = [0, 10, 20, 30, 40, 50]  # High demand
            elif predicted > 40:
                trip_minutes = [0, 15, 30, 45]  # Medium
            else:
                trip_minutes = [0, 30]  # Low
            for minute in trip_minutes:
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
            prev_demand = predicted  # Update for next hour
    app_state["optimized_schedule"] = schedule
    print("--- 📅 Optimized schedule generated based on demand predictions. ---")

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
    routes_df = app_state["routes_df"]
    num_stops = len([s for s in routes_df[routes_df['route_id'] == route_id]['stops'].iloc[0].split(';')])
    predictions = []
    prev_demand = 50.0
    try:
        df = pd.read_csv("data/historical_ticket_sales.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent = df[df['route_id'] == route_id].groupby(df['timestamp'].dt.hour)['passengers_boarded'].sum().tail(1)
        prev_demand = recent.values[0] if not recent.empty else 50.0
    except:
        pass
    for i in range(4):
        future_time = now + timedelta(hours=i)
        hour, day_of_week = future_time.hour, future_time.weekday()
        route_hour_interaction = route_id * hour
        prediction_input = pd.DataFrame(
            [[route_id, hour, day_of_week, prev_demand, num_stops, route_hour_interaction]],
            columns=['route_id', 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction']
        )
        predicted_passengers = model.predict(prediction_input)[0]
        predictions.append({"hour": hour, "predicted_total_passengers": round(predicted_passengers)})
        prev_demand = predicted_passengers
    return {"route_id": route_id, "generated_at": now.isoformat(), "predictions": predictions}

@app.post("/api/live_update")
def live_update(update: BusUpdate):
    print(f"Received update for {update.bus_id} on trip {update.trip_id}")
    update_time = datetime.fromisoformat(update.timestamp)
    app_state["live_bus_locations"][update.bus_id] = {
        "route_id": update.route_id,
        "trip_id": update.trip_id,
        "location": (update.latitude, update.longitude),
        "occupancy": update.occupancy,
        "timestamp": update_time
    }
    BUNCHING_THRESHOLD_METERS = 500
    for bus_id, data in app_state["live_bus_locations"].items():
        if bus_id != update.bus_id and data["route_id"] == update.route_id:
            loc1 = (update.latitude, update.longitude)
            loc2 = data["location"]
            distance_km = hs.haversine(loc1, loc2)
            distance_m = distance_km * 1000
            if distance_m < BUNCHING_THRESHOLD_METERS:
                print(f"ALERT 🚨: Bus bunching detected! Buses {update.bus_id} and {bus_id} are too close ({distance_m:.2f} m) on route {update.route_id}.")
                print(f"Action: Adjust schedule for bus {bus_id} to increase headway.")
    OFF_PEAK_THRESHOLD = 50
    HIGH_DEMAND_THRESHOLD = 80
    try:
        model = app_state["prediction_model"]
        now = datetime.now()
        routes_df = app_state["routes_df"]
        num_stops = len([s for s in routes_df[routes_df['route_id'] == update.route_id]['stops'].iloc[0].split(';')])
        df = pd.read_csv("data/historical_ticket_sales.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent = df[df['route_id'] == update.route_id].groupby(df['timestamp'].dt.hour)['passengers_boarded'].sum().tail(1)
        prev_demand = recent.values[0] if not recent.empty else 50.0
        route_hour_interaction = update.route_id * now.hour
        prediction_input = pd.DataFrame(
            [[update.route_id, now.hour, now.weekday(), prev_demand, num_stops, route_hour_interaction]],
            columns=['route_id', 'hour_of_day', 'day_of_week', 'prev_hour_demand', 'num_stops', 'route_hour_interaction']
        )
        predicted_passengers = model.predict(prediction_input)[0]
        print(f"Current occupancy for {update.bus_id}: {update.occupancy}. Predicted total demand: {round(predicted_passengers)} passengers.")
        if predicted_passengers < OFF_PEAK_THRESHOLD:
            print("Suggestion: Off-peak period. Consider reducing bus frequency on this route.")
        elif predicted_passengers > HIGH_DEMAND_THRESHOLD:
            print("Suggestion: High-demand period. Consider increasing bus frequency on this route.")
    except Exception as e:
        print(f"Error during demand-based logic: {e}")
    return {"status": "success", "message": f"Live update for {update.bus_id} received. Applied scheduling logic."}

@app.post("/api/retrain_model")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_demand_prediction_model)
    return {"message": "Model retraining started in the background. Check server logs for progress."}

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))  # Render sets $PORT dynamically
    uvicorn.run("main:app", host="0.0.0.0", port=port)
