# Wind Power Forecasting FastAPI App

This project is a machine learning-based application for forecasting wind power generation. It serves a predictive model through a FastAPI backend and provides a simple web interface for interaction.

## Features

- **Multiple Models**: Utilizes an ensemble of models including XGBoost, LightGBM, CatBoost, and LSTM for robust predictions.
- **FastAPI Backend**: High-performance, easy-to-use API framework.
- **Web Interface**: Simple HTML/JS frontend to input features and view predictions.
- **Real-time Prediction**: Returns power output estimates in Kilowatts (KW).

## Prerequisites

- Python 3.8+
- pip

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/muhammadjamal1155/wind-power-forecasting-notebook.git
    cd wind-power-forecasting-notebook
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Start the server:**
    ```bash
    uvicorn app:app --reload --port 8000
    ```

2.  **Access the application:**
    Open your browser and navigate to `http://127.0.0.1:8000`.

3.  **API Documentation:**
    The interactive API docs are available at `http://127.0.0.1:8000/docs`.

## Project Structure

- `app.py`: Main FastAPI application entry point.
- `models/`: Directory containing trained model files (`.pkl`, `.cbm`, `.keras`) and scalers.
- `templates/`: Contains `index.html` for the frontend.
- `requirements.txt`: List of Python dependencies.

## Inputs

The model accepts the following features for prediction:
- Wind Speed (m/s)
- Theoretical Power Curve (KWh)
- Wind Direction (°)
- Time features (Hour, Day of Week)
- Lag features (Power, Wind Speed)
- Rolling Mean features
