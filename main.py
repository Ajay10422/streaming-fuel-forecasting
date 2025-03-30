from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import qrcode
from PIL import Image
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load the trained models
try:
    co2_model = joblib.load("xgboost_model.joblib")
    combined_fuel_model = joblib.load("xgboost_model_Combined_(L_100_km).joblib")
    city_fuel_model = joblib.load("xgboost_model_City_(L_100_km).joblib")
    highway_fuel_model = joblib.load("xgboost_model_Highway_(L_100_km).joblib")
    smog_model = joblib.load("xgboost_model_Smog_rating.joblib")
    logger.info("All models loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# Load the dataset
try:
    data = pd.read_excel("C:/Users/ajayk/Downloads/Cars_leaned.xlsx")  # Update with your actual path
    logger.info("Dataset loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Dataset file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Define mappings
brand_mapping = {
    'Toyota': 'Toyota', 'Lexus': 'Toyota', 'Mazda': 'Toyota', 'Subaru': 'Toyota',
    'Volkswagen': 'Volkswagen Group', 'Audi': 'Volkswagen Group', 'Porsche': 'Volkswagen Group', 'Bentley': 'Volkswagen Group', 
    'Bugatti': 'Volkswagen Group', 'Lamborghini': 'Volkswagen Group', 
    'Chevrolet': 'General Motors', 'GMC': 'General Motors', 'Cadillac': 'General Motors', 'Buick': 'General Motors',
    'Ford': 'Ford Motor Company', 'Lincoln': 'Ford Motor Company', 'Jaguar': 'Ford Motor Company',
    'Chrysler': 'Stellantis', 'Dodge': 'Stellantis', 'Jeep': 'Stellantis', 'Ram': 'Stellantis', 
    'FIAT': 'Stellantis', 'Maserati': 'Stellantis', 'Alfa Romeo': 'Stellantis',
    'Honda': 'Honda', 'Acura': 'Honda',
    'Hyundai': 'Hyundai Motor Group', 'Kia': 'Hyundai Motor Group', 'Genesis': 'Hyundai Motor Group',
    'BMW': 'BMW Group', 'MINI': 'BMW Group', 'Rolls-Royce': 'BMW Group',
    'Mercedes-Benz': 'Mercedes-Benz Group', 'Aston Martin': 'Mercedes-Benz Group',
    'Nissan': 'Nissan-Renault Alliance', 'Infiniti': 'Nissan-Renault Alliance', 'Mitsubishi': 'Nissan-Renault Alliance',
    'Ferrari': 'Ferrari', 'Land Rover': 'Tata', 'Jaguar': 'Tata', 'Volvo': 'Volvo'
}

transmission_mapping = {
    'M5': 'Manual', 'M6': 'Manual', 'M7': 'Manual',
    'A4': 'Automatic', 'A5': 'Automatic', 'A6': 'Automatic', 'A7': 'Automatic', 'A8': 'Automatic', 'A9': 'Automatic', 'A10': 'Automatic',
    'AS5': 'Automated Manual', 'AS6': 'Automated Manual', 'AS7': 'Automated Manual', 'AS8': 'Automated Manual', 'AS9': 'Automated Manual', 'AS10': 'Automated Manual',
    'AM6': 'Dual-Clutch', 'AM7': 'Dual-Clutch', 'AM8': 'Dual-Clutch', 'AM9': 'Dual-Clutch',
    'AV': 'CVT', 'AV6': 'CVT', 'AV7': 'CVT', 'AV8': 'CVT', 'AV10': 'CVT', 'AV1': 'CVT'
}

# Apply mappings to the dataset
data_mapped = data.copy()
data_mapped['Make'] = data_mapped['Make'].map(brand_mapping)
data_mapped['Transmission'] = data_mapped['Transmission'].map(transmission_mapping)

# Define features used during training
features = ["Model year", "Engine size (L)", "Cylinders", "Combined (L/100 km)", "Combined (mpg)", 
            "CO2 rating", "Smog rating", "Make", "Transmission", "Fuel type"]
categorical_cols = ['Make', 'Transmission', 'Fuel type']

# Fit the OneHotEncoder
try:
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    data_categorical = data_mapped[categorical_cols]
    ohe.fit(data_categorical)
    expected_columns = (list(data[features].select_dtypes(include=[np.number]).columns) + 
                       list(ohe.get_feature_names_out(categorical_cols)))
    logger.info("OneHotEncoder fitted successfully")
except Exception as e:
    logger.error(f"Error fitting OneHotEncoder: {e}")
    raise

# Compute statistics and vehicle class ranges
stats = data.describe().to_dict()
vehicle_class_ranges = data.groupby("Vehicle class")["Combined (L/100 km)"].agg(["min", "max"]).to_dict()

# Define valid options for validation
makes = list(set(brand_mapping.values()))
transmissions = list(set(transmission_mapping.values()))
fuel_types = ["X", "Z", "E", "D"]

# Generate QR code
def generate_qr_code(url: str, filename: str):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(f"static/{filename}")

@app.on_event("startup")
async def startup_event():
    generate_qr_code("http://vehicles.nrcan.gc.ca", "qr-code.png")
    logger.info("QR code generated")

# Preprocessing function
def preprocess_input(vehicle_year: int, make: str, transmission: str, fuel_type: str) -> pd.DataFrame:
    logger.info(f"Preprocessing input: year={vehicle_year}, make={make}, transmission={transmission}, fuel_type={fuel_type}")
    
    # Validate inputs
    if vehicle_year < 2017 or vehicle_year > 2025:
        raise ValueError("Model year must be between 2017 and 2025")
    mapped_make = brand_mapping.get(make, make)
    mapped_transmission = transmission_mapping.get(transmission, transmission)
    if mapped_make not in makes:
        raise ValueError(f"Invalid make. Expected one of: {', '.join(makes)}")
    if mapped_transmission not in transmissions:
        raise ValueError(f"Invalid transmission. Expected one of: {', '.join(transmissions)}")
    if fuel_type not in fuel_types:
        raise ValueError(f"Invalid fuel type. Expected one of: {', '.join(fuel_types)}")

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Model year": [vehicle_year],
        "Make": [mapped_make],
        "Transmission": [mapped_transmission],
        "Fuel type": [fuel_type]
    })

    # Add default numerical features using dataset means
    input_data["Engine size (L)"] = data["Engine size (L)"].mean()
    input_data["Cylinders"] = data["Cylinders"].mean()
    input_data["Combined (L/100 km)"] = data["Combined (L/100 km)"].mean()
    input_data["Combined (mpg)"] = data["Combined (mpg)"].mean()
    input_data["CO2 rating"] = data["CO2 rating"].mean()
    input_data["Smog rating"] = data["Smog rating"].mean()

    # One-hot encode categorical variables
    try:
        input_categorical = input_data[categorical_cols]
        encoded_data = ohe.transform(input_categorical)
        input_encoded = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(categorical_cols)).astype(int)
    except Exception as e:
        logger.error(f"Error during one-hot encoding: {e}")
        raise

    # Combine numerical and encoded data
    input_encoded = pd.concat([input_data.drop(columns=categorical_cols), input_encoded], axis=1)

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match training data
    input_encoded = input_encoded[expected_columns]
    logger.info("Input preprocessing completed successfully")

    return input_encoded

# Prediction function (updated to handle NumPy types)
def get_predictions(input_data: pd.DataFrame) -> dict:
    logger.info("Generating predictions")
    try:
        # Convert NumPy float32 to Python float
        co2_prediction = float(co2_model.predict(input_data)[0])
        combined_fuel = float(combined_fuel_model.predict(input_data)[0])
        city_fuel = float(city_fuel_model.predict(input_data)[0])
        highway_fuel = float(highway_fuel_model.predict(input_data)[0])
        smog_rating = int(round(float(smog_model.predict(input_data)[0])))  # Convert to float then round to int

        # Calculate additional metrics
        combined_mpg = 235.215 / combined_fuel if combined_fuel > 0 else 0
        annual_fuel_cost = (combined_fuel * 20000 / 100) * 1.09

        predictions = {
            "prediction": round(co2_prediction, 2),
            "combined_fuel": round(combined_fuel, 1),
            "city_fuel": round(city_fuel, 1),
            "highway_fuel": round(highway_fuel, 1),
            "combined_mpg": int(round(combined_mpg)),
            "annual_fuel_cost": int(round(annual_fuel_cost)),
            "smog_rating": smog_rating
        }
        logger.info(f"Predictions generated: {predictions}")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

# Homepage
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
        "makes": makes,
        "transmissions": transmissions,
        "fuel_types": fuel_types,
        "prediction": None,
        "combined_fuel": None,
        "city_fuel": None,
        "highway_fuel": None,
        "combined_mpg": None,
        "annual_fuel_cost": None,
        "vehicle_class_range": None,
        "smog_rating": None,
        "title": "CO2 Emissions Predictor (2017-2025)"
    })

# JSON prediction endpoint
@app.post("/predict_json", response_class=JSONResponse)
async def predict_json(
    vehicle_year: int = Form(...),
    make: str = Form(...),
    transmission: str = Form(...),
    fuel_type: str = Form(...)
):
    try:
        # Preprocess input
        input_data = preprocess_input(vehicle_year, make, transmission, fuel_type)

        # Get predictions
        predictions = get_predictions(input_data)

        # Determine vehicle class and range
        mapped_make = brand_mapping.get(make, make)
        vehicle_class = data[data["Make"] == mapped_make]["Vehicle class"].mode().iloc[0] if mapped_make in data["Make"].values else "Sport utility vehicle: Small"
        vehicle_class_range = f"{vehicle_class_ranges['min'][vehicle_class]:.1f} – {vehicle_class_ranges['max'][vehicle_class]:.1f}"
        predictions["vehicle_class_range"] = vehicle_class_range

        return JSONResponse(content=predictions)
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error in predict_json: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

# Serve QR code image
@app.get("/qr-code", response_class=FileResponse)
async def get_qr_code():
    return FileResponse("static/qr-code.png")

# Run with: uvicorn main:app --reload