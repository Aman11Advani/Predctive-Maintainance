from flask import Flask, request, jsonify
import pandas as pd
from Hackathon_Hotspot_Maintenance import DeviceHotspotPredictor
from Hackathon_Next_Maintenance import DeviceNextMaintenance
from flask_cors import CORS
import logging

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize models at startup
try:
    hotspot_predictor = DeviceHotspotPredictor("final.csv")
    hotspot_predictor.train_model()
    maintenance_predictor = DeviceNextMaintenance("final.csv")
    maintenance_predictor.train_model()
    logging.info("Models initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing models: {e}")
    raise e

@app.route('/')
def index():
    return "Welcome to the PMD AI backend!"

@app.route('/hotspot-maintenance', methods=['POST'])
def hotspot_maintenance():
    try:
        data = request.json
        # Validate input keys
        required_keys = ['latitude', 'longitude', 'weather', 'wind_speed', 'rainfall']
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required field: {key}"}), 400

        # Extract and validate input values
        try:
            latitude = float(data['latitude'])
            longitude = float(data['longitude'])
            weather = int(data['weather'])
            wind_speed = float(data['wind_speed'])
            rainfall = float(data['rainfall'])
            proximity_range = float(data.get('proximity_range'))
            risk_threshold = int(data.get('risk_threshold'))
        except ValueError:
            return jsonify({"error": "Invalid data type in input fields"}), 400

        # Perform prediction
        result = hotspot_predictor.predict_hotspots(
            latitude=latitude,
            longitude=longitude,
            weather=weather,
            wind_speed=wind_speed,
            rainfall=rainfall,
            proximity_range=proximity_range,
            risk_threshold=risk_threshold
        )

        if isinstance(result, str):  # No hotspots found
            return jsonify({"message": result})

        return jsonify(result.to_dict(orient='records'))

    except Exception as e:
        logging.error(f"Error in /hotspot-maintenance: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/next-maintenance', methods=['GET'])
def next_maintenance():
    try:
        # Predict maintenance for all devices
        predictions = maintenance_predictor.predict_all_devices()
        result = predictions.to_dict(orient="records")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in /next-maintenance: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
