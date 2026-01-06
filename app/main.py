"""
FastAPI application for customer churn prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)


class CustomerFeatures(BaseModel):
    """Input features for customer churn prediction."""
    features: List[float]


class PredictionResponse(BaseModel):
    """Response model for churn prediction."""
    churn_probability: float
    churn_prediction: int
    message: str


# Global variable to store the model
model = None


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global model
    try:
        # Load your trained model here
        # model = joblib.load('models/churn_model.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """
    Predict customer churn based on input features.
    
    Args:
        customer: CustomerFeatures object containing feature values
        
    Returns:
        PredictionResponse with churn probability and prediction
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train and load a model first."
        )
    
    try:
        # Convert features to numpy array
        features_array = np.array(customer.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0][1]
        
        return PredictionResponse(
            churn_probability=float(probability),
            churn_prediction=int(prediction),
            message="Prediction successful"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
