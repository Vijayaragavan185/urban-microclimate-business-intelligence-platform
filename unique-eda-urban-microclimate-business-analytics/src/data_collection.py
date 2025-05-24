import requests
import json
from geopy.distance import geodesic
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns

class EnvironmentalDataCollector:
    """
    Professional data collection class with error handling.
    
    Key Concepts:
    - Class-based organization (reusable, maintainable)
    - Error handling (graceful failures)
    - Rate limiting (respectful API usage)
    - Data validation (quality assurance)
    """
    
    def __init__(self, api_keys):
        self.api_keys = api_keys
        
    def collect_weather_data(self, locations):
        """
        Collect weather data for multiple locations.
        
        Concepts Demonstrated:
        - API requests with parameters
        - Error handling with try/except
        - Data transformation
        - Progress tracking for user experience
        """
        weather_data = []
        
        print(f"🌤️ Collecting weather data for {len(locations)} locations...")
        
        for i, location in enumerate(locations):
            try:
                # Show progress (professional user experience)
                if i % 5 == 0:
                    print(f"   Processing location {i+1}/{len(locations)}")
                
                # OpenWeatherMap API call
                url = "http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': location['lat'],
                    'lon': location['lon'],
                    'appid': self.api_keys.get('openweather', 'demo_key'),
                    'units': 'metric'  # Celsius temperature
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract relevant weather metrics
                    weather_point = {
                        'street_name': location.get('street_name', f"Location_{i}"),
                        'latitude': location['lat'],
                        'longitude': location['lon'],
                        'temperature_celsius': data['main']['temp'],
                        'humidity_percent': data['main']['humidity'],
                        'wind_speed_ms': data.get('wind', {}).get('speed', 0),
                        'pressure_hpa': data['main']['pressure'],
                        'weather_description': data['weather'][0]['description'],
                        'timestamp': datetime.now(),
                        'data_source': 'openweathermap'
                    }
                    
                    weather_data.append(weather_point)
                    
                else:
                    print(f"   ⚠️ Weather API failed for {location.get('street_name')}: Status {response.status_code}")
                
                # Rate limiting - be respectful to the API
                time.sleep(0.1)  # 100ms delay between requests
                
            except Exception as e:
                print(f"   ❌ Error collecting weather for {location.get('street_name')}: {e}")
                continue
        
        print(f"✅ Collected weather data for {len(weather_data)} locations")
        return pd.DataFrame(weather_data)
    
    def simulate_air_quality_data(self, weather_df):
        """
        Simulate air quality data based on weather patterns.
        
        Why Simulate?
        - Air quality APIs often require paid subscriptions
        - Demonstrates realistic data relationships
        - Shows how to handle missing data sources
        
        Concepts:
        - Data simulation with realistic patterns
        - Feature relationships (wind affects air quality)
        - Random variation with constraints
        """
        print("🌬️ Simulating realistic air quality data...")
        
        air_quality_data = []
        
        for _, row in weather_df.iterrows():
            # Base AQI influenced by weather conditions
            base_aqi = 50  # Moderate baseline
            
            # Wind speed affects air quality (higher wind = cleaner air)
            wind_effect = max(0, (row['wind_speed_ms'] - 2) * -5)
            
            # Humidity effect (very high humidity can trap pollutants)
            humidity_effect = max(0, (row['humidity_percent'] - 70) * 0.3)
            
            # Add realistic random variation
            random_variation = np.random.normal(0, 15)
            
            # Calculate final AQI
            aqi = base_aqi + wind_effect + humidity_effect + random_variation
            aqi = max(10, min(200, aqi))  # Keep within realistic bounds
            
            air_quality_point = {
                'street_name': row['street_name'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'air_quality_index': int(aqi),
                'pm25_concentration': aqi * 0.4,  # Realistic PM2.5 relationship
                'timestamp': row['timestamp'],
                'data_source': 'simulated_realistic'
            }
            
            air_quality_data.append(air_quality_point)
        
        print(f"✅ Generated realistic air quality data for {len(air_quality_data)} locations")
        return pd.DataFrame(air_quality_data)

# Example usage with sample locations
sample_locations = [
    {'lat': 40.7128, 'lon': -74.0060, 'street_name': 'Times Square NYC'},
    {'lat': 40.7589, 'lon': -73.9851, 'street_name': 'Central Park NYC'},
    {'lat': 40.7505, 'lon': -73.9934, 'street_name': 'Broadway NYC'},
    {'lat': 40.7282, 'lon': -74.0776, 'street_name': 'Financial District NYC'},
    {'lat': 40.7831, 'lon': -73.9712, 'street_name': 'Upper East Side NYC'}
]

# Initialize collector (you can add real API keys later)
api_keys = {'openweather': 'your_api_key_here'}  # Replace with real key
collector = EnvironmentalDataCollector(api_keys)

# For this tutorial, let's simulate the data collection
print("📡 Simulating data collection process...")
print("(In a real project, you'd use actual API keys)")

# Create simulated environmental data
def create_simulated_environmental_data():
    """Create realistic environmental data for learning purposes"""
    
    env_data = []
    for i, location in enumerate(sample_locations):
        # Simulate realistic environmental conditions
        base_temp = 20 + np.random.normal(0, 5)  # Around 20°C with variation
        humidity = max(30, min(90, 60 + np.random.normal(0, 15)))
        wind_speed = max(0, np.random.exponential(2))  # Exponential distribution for wind
        aqi = max(20, min(150, 50 + np.random.normal(0, 20)))
        
        env_point = {
            'street_name': location['street_name'],
            'latitude': location['lat'],
            'longitude': location['lon'],
            'temperature_celsius': round(base_temp, 1),
            'humidity_percent': round(humidity, 1),
            'wind_speed_ms': round(wind_speed, 1),
            'air_quality_index': int(aqi),
            'timestamp': datetime.now(),
            'data_source': 'simulated_for_tutorial'
        }
        env_data.append(env_point)
    
    return pd.DataFrame(env_data)

# Create our environmental dataset
env_df = create_simulated_environmental_data()
print("\n📊 Environmental Data Sample:")
print(env_df.head())
