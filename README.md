# **Smart Bus Optimization System**

A real-time, data-driven platform to solve urban transit inefficiencies in Indian Tier-1 cities. This system moves beyond static timetables by leveraging a machine learning model to predict passenger demand and dynamically optimize bus schedules, aiming to reduce bus bunching, eliminate under-utilized trips, and provide a predictable, superior experience for commuters.

## **🚀 Live Demo**

*https://hackathon-i2pv.onrender.com*

## **✨ Core Features**

* **Dynamic Scheduling Engine:** Generates an optimized, demand-responsive schedule that increases bus frequency during peak hours and reduces it during off-peak times.  
* **ML-Powered Demand Prediction:** Utilizes a pre-trained **XGBoost Regressor** model to accurately forecast passenger ridership on an hourly basis for each route.  
* **Real-Time Simulation:** Simulates live bus movements and data updates, providing a dynamic view of the transit network.  
* **Bus Bunching Detection:** Implements the Haversine formula to calculate real-world distances between buses and automatically detects bunching incidents.  
* **"Before & After" Comparison:** A clear dashboard interface to visualize the tangible benefits of the optimized schedule (reduced wait times, better utilization) compared to the original static one.  
* **On-the-Fly Model Retraining:** An advanced API endpoint that allows the ML model to be retrained on new data, ensuring the system adapts over time.

## **🛠️ Technology Stack**

This project uses a modern, decoupled architecture with a Python backend and a Next.js frontend.

#### **Backend**

| Technology | Purpose |
| :---- | :---- |
| **Python 3.10+** | Core programming language |
| **FastAPI** | High-performance, asynchronous web framework for the API |
| **Uvicorn** | ASGI server to run the FastAPI application |
| **Pandas & NumPy** | Data manipulation and numerical computation |
| **Scikit-learn** | Machine learning utilities and model evaluation (K-Fold CV) |
| **XGBoost** | High-performance gradient boosting library for the demand model |

Export to Sheets

#### **Frontend**

| Technology | Purpose |
| :---- | :---- |
| **Next.js 14** | Production-grade React framework (App Router) |
| **React & TypeScript** | Building the user interface |
| **Tailwind CSS** | Utility-first CSS framework for rapid styling |
| **shadcn/ui** | A library of beautifully designed, accessible UI components |

Export to Sheets

## **🏗️ System Architecture**

The system is designed with a clear separation between the backend logic and the frontend presentation layer.

\+-------------------------+      \+--------------------------+  
|      Next.js Frontend   |      |      FastAPI Backend     |  
| (Dashboard, Map, Charts)|      | (Python, ML Model, Logic)|  
\+-------------------------+      \+--------------------------+  
           ^                                  ^  
           | REST API Calls (HTTP)            |  
           |                                  |  
\+----------v----------------------------------v----------+  
|              Data Layer (CSV Files / Database)        |  
|  (historical.csv, routes.csv, stops.csv)              |  
\+-------------------------------------------------------+

1. The **Frontend** (Next.js) provides the user interface for transit authorities.  
2. It communicates with the **Backend** (FastAPI) via a REST API to fetch schedules, predictions, and live updates.  
3. The **Backend** handles all data processing, runs the machine learning model, generates schedules, and detects operational issues.

## **⚙️ Setup and Installation**

To get this project running locally, follow these steps.

### **Prerequisites**

* Python 3.10 or higher  
* Node.js v18.17 or higher  
* `pip` and `npm` package managers

### **1\. Clone the Repository**

Bash  
git clone https://github.com/your-username/your-repository-name.git  
cd your-repository-name

### **2\. Backend Setup**

Navigate to the backend directory and set up a virtual environment.

Bash  
cd backend

\# Create a virtual environment  
python \-m venv venv

\# Activate the virtual environment  
\# On Windows:  
venv\\Scripts\\activate  
\# On macOS/Linux:  
source venv/bin/activate

\# Install the required Python packages  
pip install \-r requirements.txt

### **3\. Frontend Setup**

Navigate to the frontend source directory.

Bash  
\# From the root project directory  
cd src

\# Install the required Node.js packages  
npm install

## **▶️ Running the Application**

### **1\. Start the Backend Server**

Make sure you are in the `backend` directory with your virtual environment activated.

Bash  
uvicorn main:app \--reload

The backend API will now be running and accessible at `http://127.0.0.1:8000`. You can see the interactive API documentation at `http://127.0.0.1:8000/docs`.

### **2\. Start the Frontend Development Server**

Open a new terminal, navigate to the `src` directory.

Bash  
npm run dev

The frontend application will now be running and accessible at `http://localhost:3000`.

## **📡 API Endpoints**

The backend exposes several key endpoints for the frontend to consume:

* `GET /api/routes`: Fetches details of all bus routes and their stops.  
* `GET /api/schedule/static`: Returns the original, fixed bus schedule.  
* `GET /api/schedule/optimized`: Returns the dynamic, ML-powered optimized schedule.  
* `GET /api/predictions/{route_id}`: Provides hourly passenger demand forecasts for a specific route.  
* `POST /api/live_update`: Simulates a live data feed to update bus locations and check for bunching.  
* `POST /api/retrain_model`: Triggers a background task to retrain the ML model.

