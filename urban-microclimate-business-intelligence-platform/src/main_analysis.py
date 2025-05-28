"""
Urban Micro-Climate Business Intelligence Platform - Main Analysis Pipeline
Author: Vijayaragavan D
Description: Complete integration and execution of urban environmental-business analysis
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import our custom modules
from data_collectors import EnvironmentalDataCollector, BusinessDataCollector, main_collection_pipeline
from business_analyzer import BusinessPerformanceAnalyzer, AnalysisResults
from spatial_processor import SpatialProcessor
from feature_engineering import FeatureEngineer
from visualization_utils import VisualizationEngine

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('urban_microclimate_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class UrbanMicroClimateAnalysisPlatform:
    """
    Complete urban micro-climate business intelligence analysis platform.
    
    This class orchestrates the entire analysis pipeline from data collection
    through visualization, providing a comprehensive solution for understanding
    the relationship between environmental conditions and business performance.
    """
    
    def __init__(self, project_config: Dict = None):
        self.config = project_config or self._load_default_config()
        self.results = {}
        self.data = {}
        self.analysis_timestamp = datetime.now()
        
        # Initialize all components
        self.data_collector = None
        self.business_analyzer = BusinessPerformanceAnalyzer()
        self.spatial_processor = SpatialProcessor()
        self.feature_engineer = FeatureEngineer()
        self.viz_engine = VisualizationEngine()
        
        # Ensure output directories exist
        self._setup_output_directories()
        
        logger.info("Urban Micro-Climate Analysis Platform initialized")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration for analysis."""
        return {
            'project_name': 'Urban Micro-Climate Business Intelligence',
            'analysis_version': '1.0.0',
            'spatial_join_distance': 200,  # meters
            'cluster_distance': 300,  # meters
            'minimum_businesses_per_area': 3,
            'output_formats': ['csv', 'json', 'html'],
            'visualization_formats': ['png', 'html'],
            'statistical_significance_threshold': 0.05
        }
    
    def _setup_output_directories(self):
        """Create necessary output directories."""
        directories = [
            'results',
            'results/data',
            'results/visualizations', 
            'results/reports',
            'results/models',
            'data/processed'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Output directories configured")
    
    def run_complete_analysis(self) -> Dict:
        """Execute the complete analysis pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE URBAN MICRO-CLIMATE BUSINESS ANALYSIS")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Data Collection
            logger.info("Phase 1: Data Collection and Preparation")
            self._execute_data_collection()
            
            # Phase 2: Feature Engineering
            logger.info("Phase 2: Advanced Feature Engineering")
            self._execute_feature_engineering()
            
            # Phase 3: Spatial Analysis
            logger.info("Phase 3: Geospatial Analysis and Integration")
            self._execute_spatial_analysis()
            
            # Phase 4: Business Intelligence Analysis
            logger.info("Phase 4: Business Performance Analysis")
            self._execute_business_analysis()
            
            # Phase 5: Visualization and Reporting
            logger.info("Phase 5: Comprehensive Visualization and Reporting")
            self._execute_visualization()
            
            # Phase 6: Final Report Generation
            logger.info("Phase 6: Executive Summary and Insights")
            self._generate_final_report()
            
            logger.info("=" * 80)
            logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise
    
    def _execute_data_collection(self):
        """Execute comprehensive data collection phase."""
        logger.info("Executing data collection pipeline...")
        
        try:
            # Run main collection pipeline from data_collectors module
            env_data, business_data = main_collection_pipeline()
            
            # Store in class data structure
            self.data['environmental_raw'] = env_data
            self.data['business_raw'] = business_data
            
            # Save raw data
            env_data.to_csv('data/processed/environmental_raw.csv', index=False)
            business_data.to_csv('data/processed/business_raw.csv', index=False)
            
            # Log collection results
            logger.info(f"Environmental data collected: {len(env_data)} measurements")
            logger.info(f"Business data collected: {len(business_data)} profiles")
            
            # Store collection statistics
            self.results['data_collection'] = {
                'environmental_points': len(env_data),
                'business_profiles': len(business_data),
                'collection_timestamp': self.analysis_timestamp.isoformat(),
                'geographic_coverage': {
                    'lat_range': [env_data['latitude'].min(), env_data['latitude'].max()],
                    'lon_range': [env_data['longitude'].min(), env_data['longitude'].max()]
                }
            }
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            raise
    
    def _execute_feature_engineering(self):
        """Execute advanced feature engineering phase."""
        logger.info("Executing feature engineering pipeline...")
        
        try:
            # Apply comprehensive feature engineering
            env_featured, business_featured = self.feature_engineer.create_all_features(
                self.data['environmental_raw'],
                self.data['business_raw']
            )
            
            # Store engineered data
            self.data['environmental_featured'] = env_featured
            self.data['business_featured'] = business_featured
            
            # Save engineered data
            env_featured.to_csv('data/processed/environmental_featured.csv', index=False)
            business_featured.to_csv('data/processed/business_featured.csv', index=False)
            
            # Log feature engineering results
            env_new_features = len(env_featured.columns) - len(self.data['environmental_raw'].columns)
            business_new_features = len(business_featured.columns) - len(self.data['business_raw'].columns)
            
            logger.info(f"Environmental features created: {env_new_features} new features")
            logger.info(f"Business features created: {business_new_features} new features")
            
            # Store feature engineering statistics
            self.results['feature_engineering'] = {
                'environmental_features_added': env_new_features,
                'business_features_added': business_new_features,
                'total_environmental_features': len(env_featured.columns),
                'total_business_features': len(business_featured.columns)
            }
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _execute_spatial_analysis(self):
        """Execute geospatial analysis and integration."""
        logger.info("Executing spatial analysis pipeline...")
        
        try:
            # Spatial join of environmental and business data
            joined_data = self.spatial_processor.spatial_join(
                self.data['business_featured'],
                self.data['environmental_featured'],
                max_distance=self.config['spatial_join_distance']
            )
            
            # Store joined data
            self.data['spatially_joined'] = joined_data
            joined_data.to_csv('data/processed/spatially_joined.csv', index=False)
            
            # Create spatial clusters
            combined_points = pd.concat([
                self.data['environmental_featured'][['latitude', 'longitude']],
                self.data['business_featured'][['latitude', 'longitude']]
            ], ignore_index=True)
            
            clustered_points, cluster_summary = self.spatial_processor.create_spatial_clusters(
                combined_points,
                cluster_distance=self.config['cluster_distance']
            )
            
            # Store spatial analysis results
            self.data['spatial_clusters'] = clustered_points
            self.data['cluster_summary'] = cluster_summary
            
            logger.info(f"Spatial join completed: {len(joined_data)} business-environment pairs")
            logger.info(f"Spatial clustering: {clustered_points['cluster_id'].nunique()} clusters identified")
            
            # Store spatial analysis statistics
            self.results['spatial_analysis'] = {
                'joined_pairs': len(joined_data),
                'spatial_join_distance': self.config['spatial_join_distance'],
                'clusters_identified': int(clustered_points['cluster_id'].nunique()),
                'cluster_distance': self.config['cluster_distance']
            }
            
        except Exception as e:
            logger.error(f"Spatial analysis failed: {e}")
            raise
    
    def _execute_business_analysis(self):
        """Execute comprehensive business performance analysis."""
        logger.info("Executing business intelligence analysis...")
        
        try:
            # Load data into business analyzer
            self.business_analyzer.env_data = self.data['environmental_featured']
            self.business_analyzer.business_data = self.data['business_featured']
            
            # Use spatially joined data for analysis
            self.business_analyzer.merged_data = self._prepare_merged_data_for_analysis()
            
            # Run comprehensive analysis
            correlation_matrix = self.business_analyzer.calculate_correlation_matrix()
            statistical_tests = self.business_analyzer.perform_statistical_significance_tests()
            cluster_analysis = self.business_analyzer.perform_cluster_analysis()
            business_insights = self.business_analyzer.generate_business_insights()
            
            # Store analysis results
            self.results['business_analysis'] = {
                'correlation_matrix': correlation_matrix,
                'statistical_tests': statistical_tests,
                'cluster_analysis': cluster_analysis,
                'business_insights': business_insights
            }
            
            # Save analysis results
            correlation_matrix.to_csv('results/data/correlation_matrix.csv')
            
            with open('results/data/statistical_tests.json', 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_safe_tests = {}
                for key, value in statistical_tests.items():
                    json_safe_tests[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                          for k, v in value.items()}
                json.dump(json_safe_tests, f, indent=2)
            
            with open('results/data/business_insights.json', 'w') as f:
                json.dump(business_insights, f, indent=2, default=str)
            
            logger.info("Business analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Business analysis failed: {e}")
            raise
    
