import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from geopy.distance import geodesic
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

class DeviceHotspotPredictor:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        self.data = pd.read_csv(data_path)
        self.data['last_maintainance'] = pd.to_datetime(self.data['last_maintainance'], format='%Y%m%d%H%M%SUT')
        self.model = None
        self.scaler = StandardScaler()
        self.features = ['latitude', 'longitude', 'weather', 'wind_speed', 'rainfall', 'device_age', 'load', 'risk_factor']
        self.target = 'device_failure'
        print(f"Successfully loaded dataset with {len(self.data)} records")

    def calculate_risk_factor(self, row, conditions=None):
        """Calculate the risk factor based on multiple criteria"""
        base_risk = 0

        # Weather-based risk
        if row['wind_speed'] >= 20:
            base_risk += 2
        if row['rainfall'] >= 50:
            base_risk += 3
        if row['weather'] == 5:  # Severe weather
            base_risk += 5

        # Device-specific risk
        device_risk = {
            'Transformer': 3,
            'Feeder': 2,
            'Switch': 2,
            'Recloser': 1
        }
        base_risk += device_risk.get(row['device_type'], 0)

        # Add device age-based risk
        if row['device_age'] > 5:
            base_risk += 2
        
        # Add load-based risk
        if row['load'] > 80:
            base_risk += 1

        return base_risk

    def preprocess_data(self):
        """Preprocess the data including scaling and handling imbalanced data"""
        X = self.data.copy()

        # Calculate risk factors for each row
        X['risk_factor'] = X.apply(self.calculate_risk_factor, axis=1)

        y = X[self.target]
        X = X[self.features]

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.features)

        # Handle imbalanced data
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y)

        print("Data preprocessing completed")
        return X_res, y_res

    def train_model(self):
        """Train the Random Forest model"""
        try:
            X_res, y_res = self.preprocess_data()

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42
            )

            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train, y_train)

        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise

    def predict_hotspots(self, latitude, longitude, weather, wind_speed, rainfall,
                         proximity_range=15.0, risk_threshold=6):
        """Predict device hotspots based on weather conditions and location"""
        try:
            conditions = {
                'latitude': latitude,
                'longitude': longitude,
                'weather': weather,
                'wind_speed': wind_speed,
                'rainfall': rainfall
            }

            # Validate inputs
            if not all(isinstance(val, (int, float)) for val in conditions.values()):
                raise ValueError("All input parameters must be numeric")

            # Filter devices within proximity range
            proximity_devices = self.data[
                (self.data['latitude'].between(latitude - proximity_range, latitude + proximity_range)) &
                (self.data['longitude'].between(longitude - proximity_range, longitude + proximity_range))
            ].copy()

            if proximity_devices.empty:
                return "No devices found in the specified range."

            # Calculate risk factors for each device
            proximity_devices['risk_factor'] = proximity_devices.apply(
                lambda row: self.calculate_risk_factor(row), axis=1
            )

            # Filter and sort results
            at_risk_devices = proximity_devices[proximity_devices['risk_factor'] >= risk_threshold].sort_values(
                'risk_factor', ascending=False
            )

            if at_risk_devices.empty:
                return "No devices at high risk found."

            # Predict next maintenance dates for at-risk devices
            at_risk_devices = self.predict_next_maintenance(at_risk_devices)

            # Save results to an Excel file
            at_risk_devices.to_excel("Device_Hotspot_and_Maintenance_Results.xlsx", index=False)

            return at_risk_devices

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

    def predict_next_maintenance(self, devices):
        """Predict the next maintenance date for the given devices"""
        devices['outage_duration'] = devices['outage_duration'] / 24
        devices['next_maintenance_days'] = devices['maintenance_cycle'] - devices['outage_duration']

        # Adjust maintenance days based on device age
        devices['next_maintenance_days'] -= devices['device_age'].apply(lambda x: 10 if x > 5 else 0)

        # Adjust maintenance days based on load
        devices['next_maintenance_days'] -= devices['load'].apply(lambda x: 5 if x > 80 else 0)

        # Adjust maintenance days based on risk factor
        devices['next_maintenance_days'] -= devices['risk_factor'].apply(lambda x: x * 2)

        devices['risk_factor'] = devices['risk_factor'] / 2

        # Clip values to ensure maintenance is scheduled within a valid range
        devices['next_maintenance_days'] = devices['next_maintenance_days'].clip(lower=30, upper=180)

        # Add maintenance date prediction
        devices['predicted_next_maintenance_date'] = devices['last_maintainance'] + devices['next_maintenance_days'].apply(
            lambda x: timedelta(days=x)
        )

        return devices

def main():
    try:
        # Initialize predictor with specific file path
        predictor = DeviceHotspotPredictor("final.csv")
        predictor.train_model()

        # Get user input
        latitude = float(input("Enter latitude: "))
        longitude = float(input("Enter longitude: "))
        weather = int(input("Enter weather severity (1-5): "))
        wind_speed = float(input("Enter wind speed: "))
        rainfall = float(input("Enter rainfall: "))
        proximity_range = float(input("Enter proximity range (default 15.0): ") or 15.0)
        risk_threshold = int(input("Enter risk threshold (default 6): ") or 6)

        # Predict hotspots and maintenance dates
        print("\nPredicting hotspots for given weather conditions...")
        result = predictor.predict_hotspots(
            latitude=latitude,
            longitude=longitude,
            weather=weather,
            wind_speed=wind_speed,
            rainfall=rainfall,
            proximity_range=proximity_range,
            risk_threshold=risk_threshold
        )

        if isinstance(result, str):
            print(result)
        else:
            pd.set_option('display.max_columns', None)
            print(result)
            print("\nResults have been saved to 'Device_Hotspot_and_Maintenance_Results.xlsx'")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
