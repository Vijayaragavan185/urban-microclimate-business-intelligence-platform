# Create Module 2: business_analyzer.py with real functionality

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResults:
    correlation_matrix: pd.DataFrame
    statistical_tests: Dict
    cluster_analysis: Dict
    performance_metrics: Dict
    business_insights: Dict

class BusinessPerformanceAnalyzer:
    
    def __init__(self, database_path: str = "data/microclimate.db"):
        self.database_path = database_path
        self.env_data = None
        self.business_data = None
        self.merged_data = None
        self.analysis_results = {}
        logger.info("Business Performance Analyzer initialized")

    def load_data_from_database(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load environmental and business data from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            
            # Load environmental data
            env_query = """
            SELECT * FROM environmental_data 
            ORDER BY timestamp DESC
            """
            self.env_data = pd.read_sql_query(env_query, conn)
            
            # Load business data (simulated table for demo)
            # In real implementation, this would come from actual business data table
            conn.close()
            
            logger.info(f"Loaded {len(self.env_data)} environmental records from database")
            return self.env_data, self.business_data
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise
    
    def load_data_from_csv(self, env_file: str, business_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from CSV files."""
        try:
            self.env_data = pd.read_csv(env_file)
            self.business_data = pd.read_csv(business_file)
            
            logger.info(f"Loaded {len(self.env_data)} environmental and {len(self.business_data)} business records")
            return self.env_data, self.business_data
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise

    
    def spatial_merge_datasets(self, max_distance: float = 200.0) -> pd.DataFrame:
        """Perform spatial join between environmental and business data."""
        if self.env_data is None or self.business_data is None:
            raise ValueError("Data not loaded. Call load_data_from_csv or load_data_from_database first.")
        
        logger.info(f"Performing spatial merge with max distance: {max_distance}m")
        
        merged_records = []
        
        for _, business in self.business_data.iterrows():
            # Calculate distances to all environmental points
            distances = []
            for _, env_point in self.env_data.iterrows():
                # Simplified distance calculation (in practice, use proper geospatial functions)
                lat_diff = business['latitude'] - env_point['latitude']
                lon_diff = business['longitude'] - env_point['longitude']
                
                # Convert to approximate meters (rough calculation)
                lat_meters = lat_diff * 111000
                lon_meters = lon_diff * 111000 * np.cos(np.radians(env_point['latitude']))
                distance = np.sqrt(lat_meters**2 + lon_meters**2)
                
                distances.append(distance)
            
            # Find closest environmental point
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            if min_distance <= max_distance:
                env_record = self.env_data.iloc[min_distance_idx]
                
                # Merge records
                merged_record = {
                    # Business data
                    'business_id': business.get('business_id'),
                    'business_name': business.get('name'),
                    'business_category': business.get('category'),
                    'business_rating': business.get('rating'),
                    'business_reviews': business.get('review_count'),
                    'business_success_score': business.get('success_score'),
                    'business_price_level': business.get('price_level'),
                    'business_is_open': business.get('is_open'),
                    
                    # Environmental data
                    'env_temperature': env_record.get('temperature_celsius'),
                    'env_humidity': env_record.get('humidity_percent'),
                    'env_air_quality': env_record.get('air_quality_index'),
                    'env_wind_speed': env_record.get('wind_speed_ms'),
                    'env_comfort_index': env_record.get('comfort_index'),
                    'env_quality_score': env_record.get('quality_score'),
                    
                    # Spatial relationship
                    'distance_to_env_point': min_distance,
                    'measurement_area': env_record.get('street_name'),
                    
                    # Location data
                    'latitude': business.get('latitude'),
                    'longitude': business.get('longitude')
                }
                
                merged_records.append(merged_record)
        
        self.merged_data = pd.DataFrame(merged_records)
        logger.info(f"Spatial merge completed: {len(self.merged_data)} business-environment pairs")
        
        return self.merged_data
