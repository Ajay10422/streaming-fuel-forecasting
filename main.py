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
    co2_model = joblib.load("xgboost_model_CO2_emissions_(g_km).joblib")
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
    data = pd.read_excel("Cars_leaned.xlsx")
    logger.info("Dataset loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Dataset file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Define fuel prices (in CAD/L)
fuel_prices = {
    "D": 1.702,  # Diesel
    "E": 1.601,  # E85
    "X": 1.549,  # Regular Gasoline
    "Z": 1.841   # Premium Gasoline
}

# Define constants
ANNUAL_DISTANCE_KM = 20000
CARBON_TAX_RATE = 0.000095
PASSENGERS = 5

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
numerical_cols = ["Model year", "Engine size (L)", "Cylinders"]
categorical_cols = ['Make', 'Transmission', 'Fuel type']

# Fit the OneHotEncoder
try:
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    data_categorical = data_mapped[categorical_cols]
    ohe.fit(data_categorical)
    expected_columns = numerical_cols + list(ohe.get_feature_names_out(categorical_cols))
    logger.info("OneHotEncoder fitted successfully")
except Exception as e:
    logger.error(f"Error fitting OneHotEncoder: {e}")
    raise

# Compute vehicle class ranges
vehicle_class_ranges = data.groupby("Vehicle class")["Combined (L/100 km)"].agg(["min", "max"]).to_dict()

# Define valid options for dropdowns
makes = list(set(brand_mapping.values()))
transmissions = list(set(transmission_mapping.values()))
fuel_types = ["D", "E", "X", "Z"]

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

def preprocess_input(vehicle_year: int, make: str, transmission: str, fuel_type: str) -> pd.DataFrame:
    logger.info(f"Preprocessing input: year={vehicle_year}, make={make}, transmission={transmission}, fuel_type={fuel_type}")
    
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

    filtered_data = data_mapped[
        (data_mapped['Model year'] == vehicle_year) &
        (data_mapped['Make'] == mapped_make) &
        (data_mapped['Transmission'] == mapped_transmission) &
        (data_mapped['Fuel type'] == fuel_type)
    ]

    if not filtered_data.empty:
        engine_size = filtered_data["Engine size (L)"].mean()
        cylinders = filtered_data["Cylinders"].mean()
    else:
        engine_size = data["Engine size (L)"].mean()
        cylinders = data["Cylinders"].mean()
        logger.warning("No similar vehicles found, using overall means")

    input_data = pd.DataFrame({
        "Model year": [vehicle_year],
        "Engine size (L)": [engine_size],
        "Cylinders": [cylinders],
        "Make": [mapped_make],
        "Transmission": [mapped_transmission],
        "Fuel type": [fuel_type]
    })

    try:
        input_categorical = input_data[categorical_cols]
        encoded_data = ohe.transform(input_categorical)
        input_encoded = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(categorical_cols)).astype(int)
    except Exception as e:
        logger.error(f"Error during one-hot encoding: {e}")
        raise

    input_final = pd.concat([input_data[numerical_cols], input_encoded], axis=1)
    for col in expected_columns:
        if col not in input_final.columns:
            input_final[col] = 0
    input_final = input_final[expected_columns]
    logger.info("Input preprocessing completed successfully")
    return input_final

def get_predictions(input_data: pd.DataFrame) -> dict:
    logger.info("Generating predictions")
    try:
        co2_prediction = float(co2_model.predict(input_data)[0])
        combined_fuel = float(combined_fuel_model.predict(input_data)[0])
        city_fuel = float(city_fuel_model.predict(input_data)[0])
        highway_fuel = float(highway_fuel_model.predict(input_data)[0])
        smog_rating = int(round(float(smog_model.predict(input_data)[0])))
        combined_mpg = 235.215 / combined_fuel if combined_fuel > 0 else 0

        predictions = {
            "prediction": round(co2_prediction, 2),
            "combined_fuel": round(combined_fuel, 1),
            "city_fuel": round(city_fuel, 1),
            "highway_fuel": round(highway_fuel, 1),
            "combined_mpg": int(round(combined_mpg)),
            "smog_rating": smog_rating
        }
        logger.info(f"Predictions generated: {predictions}")
        return predictions
    except Exception as e:
        logger.error(f"Error in predictions: {e}")
        raise

# New endpoint for emission insights
@app.get("/emission_insights", response_class=JSONResponse)
async def get_emission_insights():
    try:
        # Ensure numeric data
        data['Engine size (L)'] = pd.to_numeric(data['Engine size (L)'], errors='coerce')

        # Group by Fuel Type
        fuel_emissions = data.groupby('Fuel type')['CO2 emissions (g/km)'].mean().sort_values()
        fuel_data = [{"fuel_type": k, "avg_co2": round(v, 2)} for k, v in fuel_emissions.items()]

        # Group by Transmission Type
        transmission_emissions = data.groupby('Transmission')['CO2 emissions (g/km)'].mean().sort_values()
        transmission_data = [{"transmission": k, "avg_co2": round(v, 2)} for k, v in transmission_emissions.items()]

        # Group by Engine Size (for boxplot-like data, we'll use min, max, mean)
        engine_emissions = data.groupby('Engine size (L)')['CO2 emissions (g/km)'].agg(['mean', 'min', 'max']).reset_index()
        engine_data = [
            {"engine_size": row['Engine size (L)'], "mean_co2": round(row['mean'], 2), "min_co2": round(row['min'], 2), "max_co2": round(row['max'], 2)}
            for _, row in engine_emissions.iterrows()
        ]

        # Top 5 Unique Best Combinations
        top_5_combinations = (
            data.sort_values(by='CO2 emissions (g/km)')
            [['Engine size (L)', 'Cylinders', 'Transmission', 'Fuel type', 'CO2 emissions (g/km)']]
            .drop_duplicates(subset=['Engine size (L)', 'Cylinders', 'Transmission', 'Fuel type'])
            .head(5)
        )
        top_5_data = [
            {
                "engine_size": row['Engine size (L)'],
                "cylinders": int(row['Cylinders']),
                "transmission": row['Transmission'],
                "fuel_type": row['Fuel type'],
                "co2_emissions": round(row['CO2 emissions (g/km)'], 2)
            }
            for _, row in top_5_combinations.iterrows()
        ]

        return JSONResponse(content={
            "fuel_emissions": fuel_data,
            "transmission_emissions": transmission_data,
            "engine_emissions": engine_data,
            "top_5_combinations": top_5_data
        })
    except Exception as e:
        logger.error(f"Error in get_emission_insights: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
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

@app.post("/predict_json", response_class=JSONResponse)
async def predict_json(
    vehicle_year: int = Form(...),
    make: str = Form(...),
    transmission: str = Form(...),
    fuel_type: str = Form(...)
):
    try:
        input_data = preprocess_input(vehicle_year, make, transmission, fuel_type)
        predictions = get_predictions(input_data)

        fuel_price = fuel_prices.get(fuel_type, 1.549)
        annual_fuel_cost = (predictions["combined_fuel"] * ANNUAL_DISTANCE_KM / 100) * fuel_price
        predictions["annual_fuel_cost"] = int(round(annual_fuel_cost))

        vehicle_class = data[data["Make"] == make]["Vehicle class"].mode().iloc[0] if make in data["Make"].values else "Sport utility vehicle: Small"
        predictions["vehicle_class_range"] = f"{vehicle_class_ranges['min'][vehicle_class]:.1f} – {vehicle_class_ranges['max'][vehicle_class]:.1f}"

        co2_rating = max(1, min(10, 11 - (predictions["prediction"] / 50)))
        eco_score = (co2_rating + predictions["smog_rating"]) / 2
        co2_tax = predictions["prediction"] * ANNUAL_DISTANCE_KM * CARBON_TAX_RATE
        fuel_efficiency_score = 1 / (predictions["combined_fuel"] + (predictions["prediction"] / 100)) if predictions["combined_fuel"] > 0 else 0
        cost_per_passenger_km = (fuel_price / predictions["combined_fuel"]) * (100 / PASSENGERS) if predictions["combined_fuel"] > 0 else 0

        predictions["co2_rating"] = round(co2_rating, 1)
        predictions["eco_score"] = round(eco_score, 1)
        predictions["co2_tax"] = round(co2_tax, 2)
        predictions["fuel_efficiency_score"] = round(fuel_efficiency_score, 4)
        predictions["cost_per_passenger_km"] = round(cost_per_passenger_km, 4)

        logger.info(f"Returning predictions: {predictions}")
        return JSONResponse(content=predictions)
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error in predict_json: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/qr-code", response_class=FileResponse)
async def get_qr_code():
    return FileResponse("static/qr-code.png")
