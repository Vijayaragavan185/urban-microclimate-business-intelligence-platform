import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_definitions = {}
        logger.info("Feature engineer initialized")

    def create_environmental_comfort_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive environmental comfort index."""
        df_featured = df.copy()
        
        def calculate_comfort_score(row):
            # Temperature comfort (optimal: 20°C)
            temp_score = max(0, 1 - abs(row.get('temperature_celsius', 20) - 20) / 15)
            
            # Humidity comfort (optimal: 50%)
            humidity_score = max(0, 1 - abs(row.get('humidity_percent', 50) - 50) / 40)
            
            # Air quality comfort (lower AQI better)
            aqi_score = max(0, 1 - row.get('air_quality_index', 50) / 150)
            
            # Wind comfort (optimal: 2 m/s)
            wind_score = max(0, 1 - abs(row.get('wind_speed_ms', 2) - 2) / 5)
            
            # Noise comfort (if available)
            if 'noise_level_db' in row:
                noise_score = max(0, 1 - max(0, row['noise_level_db'] - 55) / 30)
            else:
                noise_score = 0.7  # Default
            
            # Weighted composite
            comfort_index = (
                temp_score * 0.25 +
                aqi_score * 0.25 +
                humidity_score * 0.20 +
                wind_score * 0.15 +
                noise_score * 0.15
            )
            
            return round(comfort_index, 3)
        
        df_featured['environmental_comfort_index'] = df.apply(calculate_comfort_score, axis=1)
        
        # Create comfort categories
        df_featured['comfort_category'] = pd.cut(
            df_featured['environmental_comfort_index'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Poor', 'Fair', 'Good', 'Excellent'],
            include_lowest=True
        )
        
        logger.info("Environmental comfort index created")
        return df_featured
