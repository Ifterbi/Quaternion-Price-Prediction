# Quaternion Price Predictor

A machine learning project using Quaternion Mathematics to predict asset prices and momentum.

## Installation

### 1. Python Backend
Install the Python dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. React Frontend (NPM)
The frontend requires Node.js and NPM. Navigate to the `web/` directory and install the dependencies:
```bash
cd web
npm install
```

## Running the Application

1. **Start the API Server**
```bash
python main.py
```
This will start the FastAPI backend on `http://127.0.0.1:8000`.

2. **Start the Frontend Dashboard**
Open a new terminal and run:
```bash
cd web
npm run dev
```
Navigate to the provided localhost URL (e.g., `http://localhost:5173`) to view the dashboard!
