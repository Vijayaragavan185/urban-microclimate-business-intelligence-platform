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
    
    def create_business_success_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive business success score."""
        df_featured = df.copy()
        
        def calculate_success_score(row):
            # Rating component (normalized 1-5 to 0-1)
            rating_score = (row.get('rating', 3.5) - 1) / 4
            
            # Review volume component (log-normalized)
            reviews = row.get('review_count', 10)
            review_score = np.log1p(reviews) / np.log1p(300)  # Normalize to 300 reviews
            
            # Operational status
            operational_score = 1.0 if row.get('is_open', True) else 0.3
            
            # Price positioning (mid-range often optimal)
            price_level = row.get('price_level', 2)
            price_score = 1.0 - abs(price_level - 2.5) / 2.5
            
            # Composite score
            success_score = (
                rating_score * 0.4 +
                review_score * 0.3 +
                operational_score * 0.2 +
                price_score * 0.1
            )
            
            return round(min(1.0, success_score), 3)
        
        df_featured['business_success_score'] = df.apply(calculate_success_score, axis=1)
        
        # Create performance categories
        df_featured['performance_category'] = pd.cut(
            df_featured['business_success_score'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Poor', 'Average', 'Good', 'Excellent'],
            include_lowest=True
        )
        
        logger.info("Business success score created")
        return df_featured
    
    def create_temporal_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Extract temporal features from timestamp data."""
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column {timestamp_col} not found")
            return df
        
        df_featured = df.copy()
        
        # Convert to datetime if not already
        df_featured[timestamp_col] = pd.to_datetime(df_featured[timestamp_col])
        
        # Extract temporal components
        df_featured['hour'] = df_featured[timestamp_col].dt.hour
        df_featured['day_of_week'] = df_featured[timestamp_col].dt.dayofweek
        df_featured['month'] = df_featured[timestamp_col].dt.month
        df_featured['quarter'] = df_featured[timestamp_col].dt.quarter
        df_featured['is_weekend'] = (df_featured['day_of_week'] >= 5).astype(int)
        
        # Business hours indicator
        df_featured['is_business_hours'] = (
            (df_featured['hour'] >= 9) & (df_featured['hour'] <= 17) & 
            (df_featured['day_of_week'] < 5)
        ).astype(int)
        
        # Rush hour indicators
        df_featured['is_morning_rush'] = (
            (df_featured['hour'] >= 7) & (df_featured['hour'] <= 9) & 
            (df_featured['day_of_week'] < 5)
        ).astype(int)
        
        df_featured['is_evening_rush'] = (
            (df_featured['hour'] >= 17) & (df_featured['hour'] <= 19) & 
            (df_featured['day_of_week'] < 5)
        ).astype(int)
        
        logger.info("Temporal features created")
        return df_featured
    
