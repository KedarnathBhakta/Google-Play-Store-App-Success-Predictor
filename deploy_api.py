from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
from src.predictor import AppSuccessPredictor

app = FastAPI(title="Google Play Store App Prediction API")

# Load predictor and models at startup
predictor = AppSuccessPredictor()
model_files = {
    'Rating': 'models/Rating_model.joblib',
    'Popularity_Score': 'models/Popularity_Score_model.joblib',
    'Success_Binary': 'models/Success_Binary_model.joblib',
    'High_Rating_Binary': 'models/High_Rating_Binary_model.joblib'
}
for name, path in model_files.items():
    if os.path.exists(path):
        pass  # Models are loaded internally by AppSuccessPredictor

class AppFeatures(BaseModel):
    App: str = Field(..., example="My New App")
    Category: str = Field(..., example="ART_AND_DESIGN")
    Rating: Optional[float] = Field(4.0, example=4.0)
    Reviews: int = Field(..., example=100)
    Size: str = Field(..., example="25M")
    Installs: str = Field(..., example="1,000+")
    Type: str = Field(..., example="Free")
    Price: str = Field(..., example="$0")
    Content_Rating: str = Field(..., alias="Content Rating", example="Everyone")
    Genres: str = Field(..., example="ART_AND_DESIGN")
    Last_Updated: str = Field(..., alias="Last Updated", example="2024-01-01")
    Current_Ver: str = Field(..., alias="Current Ver", example="1.0")
    Android_Ver: str = Field(..., alias="Android Ver", example="4.1 and up")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "App": "My New App",
                "Category": "ART_AND_DESIGN",
                "Rating": 4.0,
                "Reviews": 100,
                "Size": "25M",
                "Installs": "1,000+",
                "Type": "Free",
                "Price": "$0",
                "Content Rating": "Everyone",
                "Genres": "ART_AND_DESIGN",
                "Last Updated": "2024-01-01",
                "Current Ver": "1.0",
                "Android Ver": "4.1 and up"
            }
        }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_app(features: AppFeatures):
    app_features = features.dict(by_alias=True)
    try:
        predictions = predictor.predict_new_app(app_features)
        if not predictions or all(v is None for v in predictions.values()):
            import traceback
            error_msg = f"Prediction failed. Input: {app_features}"
            print(error_msg)
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=error_msg)
        return {"predictions": predictions}
    except Exception as e:
        import traceback
        print(f"Exception during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    uvicorn.run("deploy_api:app", host="0.0.0.0", port=8000, reload=True) 