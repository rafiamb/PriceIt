from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# load model and metadata
model = joblib.load("model/priceit_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
neighborhoods = joblib.load("model/neighborhoods.pkl")
property_types = joblib.load("model/property_types.pkl")

# serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/metadata")
def metadata():
    return {
        "neighborhoods": neighborhoods,
        "property_types": property_types,
        "amenities": ["pool", "hot_tub", "air_conditioning", "washer", "dryer", "gym", "balcony", "kitchen", "wifi", "lake_access", "city_skyline_view"]
    }

class ListingInput(BaseModel):
    neighbourhood: str
    property_type: str
    bedrooms: float
    bathrooms: float
    accommodates: float
    minimum_nights: float
    instant_bookable: int
    amenities: list[str]

@app.post("/predict")
def predict(listing: ListingInput):
    # build a zeroed feature vector
    features = {col: 0 for col in feature_columns}

    # numeric features
    features["bedrooms"] = listing.bedrooms
    features["bathrooms"] = listing.bathrooms
    features["accommodates"] = listing.accommodates
    features["minimum_nights"] = listing.minimum_nights
    features["instant_bookable_encoded"] = listing.instant_bookable
    features["amenities_count"] = len(listing.amenities)
    features["bathrooms_per_guest"] = listing.bathrooms / max(listing.accommodates, 1)
    features["guests_per_night"] = listing.accommodates / max(listing.minimum_nights, 1)

    # guests per bedroom
    features["guests_per_bedroom"] = listing.accommodates / max(listing.bedrooms, 1) if "guests_per_bedroom" in features else 0

    # amenity flags
    for amenity in listing.amenities:
        if amenity in features:
            features[amenity] = 1

    # neighbourhood one-hot
    neighbourhood_col = f"neighbourhood_cleansed_{listing.neighbourhood}"
    if neighbourhood_col in features:
        features[neighbourhood_col] = 1

    # property type one-hot
    property_col = f"property_type_cleaned_{listing.property_type}"
    if property_col in features:
        features[property_col] = 1

    # predict
    X = np.array([[features[col] for col in feature_columns]])
    log_price = model.predict(X)[0]
    price = np.expm1(log_price)

    # confidence range ± ~$24 median error
    return {
        "predicted_price": round(float(price), 2),
        "range_low": round(float(price) - 24, 2),
        "range_high": round(float(price) + 24, 2)
    }
