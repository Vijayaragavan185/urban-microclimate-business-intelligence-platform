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
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate comprehensive correlation matrix between environmental and business factors."""
        if self.merged_data is None:
            raise ValueError("Merged data not available. Run spatial_merge_datasets first.")
        
        # Select numeric columns for correlation analysis
        correlation_columns = [
            'env_temperature', 'env_humidity', 'env_air_quality', 'env_wind_speed', 'env_comfort_index',
            'business_rating', 'business_reviews', 'business_success_score', 'business_price_level'
        ]
        
        # Filter for available columns
        available_columns = [col for col in correlation_columns if col in self.merged_data.columns]
        correlation_data = self.merged_data[available_columns].copy()
        
        # Handle missing values
        correlation_data = correlation_data.dropna()
        
        # Calculate correlation matrix
        correlation_matrix = correlation_data.corr()
        
        logger.info(f"Correlation matrix calculated for {len(available_columns)} variables")
        return correlation_matrix
    
    def perform_statistical_significance_tests(self) -> Dict:
        """Perform statistical tests for key relationships."""
        if self.merged_data is None:
            raise ValueError("Merged data not available.")
        
        test_results = {}
        
        # Test 1: Environmental comfort vs Business success
        if 'env_comfort_index' in self.merged_data.columns and 'business_success_score' in self.merged_data.columns:
            comfort_data = self.merged_data['env_comfort_index'].dropna()
            success_data = self.merged_data['business_success_score'].dropna()
            
            if len(comfort_data) > 5 and len(success_data) > 5:
                # Align the data
                aligned_data = self.merged_data[['env_comfort_index', 'business_success_score']].dropna()
                
                if len(aligned_data) > 5:
                    correlation, p_value = stats.pearsonr(
                        aligned_data['env_comfort_index'], 
                        aligned_data['business_success_score']
                    )
                    
                    test_results['comfort_vs_success'] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'sample_size': len(aligned_data),
                        'interpretation': self._interpret_correlation(correlation, p_value)
                    }
        
        # Test 2: Air quality vs Business rating
        if 'env_air_quality' in self.merged_data.columns and 'business_rating' in self.merged_data.columns:
            aligned_data = self.merged_data[['env_air_quality', 'business_rating']].dropna()
            
            if len(aligned_data) > 5:
                correlation, p_value = stats.pearsonr(
                    aligned_data['env_air_quality'], 
                    aligned_data['business_rating']
                )
                
                test_results['air_quality_vs_rating'] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'sample_size': len(aligned_data),
                    'interpretation': self._interpret_correlation(correlation, p_value)
                }
        
        # Test 3: Temperature vs Business performance
        if 'env_temperature' in self.merged_data.columns and 'business_success_score' in self.merged_data.columns:
            aligned_data = self.merged_data[['env_temperature', 'business_success_score']].dropna()
            
            if len(aligned_data) > 5:
                correlation, p_value = stats.pearsonr(
                    aligned_data['env_temperature'], 
                    aligned_data['business_success_score']
                )
                
                test_results['temperature_vs_performance'] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'sample_size': len(aligned_data),
                    'interpretation': self._interpret_correlation(correlation, p_value)
                }
        
        # Test 4: ANOVA for business categories vs environmental comfort
        if 'business_category' in self.merged_data.columns and 'env_comfort_index' in self.merged_data.columns:
            category_data = self.merged_data[['business_category', 'env_comfort_index']].dropna()
            
            if len(category_data) > 10:
                categories = category_data['business_category'].unique()
                if len(categories) >= 2:
                    category_groups = [
                        category_data[category_data['business_category'] == cat]['env_comfort_index'].values
                        for cat in categories
                    ]
                    
                    # Filter out empty groups
                    category_groups = [group for group in category_groups if len(group) > 0]
                    
                    if len(category_groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*category_groups)
                        
                        test_results['category_vs_comfort_anova'] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'categories_tested': len(category_groups),
                            'interpretation': 'Significant differences between business categories in environmental comfort' if p_value < 0.05 else 'No significant differences found'
                        }
        
        logger.info(f"Statistical tests completed: {len(test_results)} tests performed")
        return test_results
