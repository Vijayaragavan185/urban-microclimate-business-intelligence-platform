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
    
    def _prepare_merged_data_for_analysis(self) -> pd.DataFrame:
        """Prepare merged dataset for business analysis."""
        if 'spatially_joined' in self.data:
            # Use spatially joined data
            merged_data = self.data['spatially_joined'].copy()
            
            # Rename columns to match expected format
            column_mapping = {
                'left_rating': 'business_rating',
                'left_review_count': 'business_reviews', 
                'left_success_score': 'business_success_score',
                'left_category': 'business_category',
                'left_name': 'business_name',
                'left_price_level': 'business_price_level',
                'left_is_open': 'business_is_open',
                'right_temperature_celsius': 'env_temperature',
                'right_humidity_percent': 'env_humidity',
                'right_air_quality_index': 'env_air_quality',
                'right_wind_speed_ms': 'env_wind_speed',
                'right_comfort_index': 'env_comfort_index',
                'right_quality_score': 'env_quality_score'
            }
            
            # Rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in merged_data.columns:
                    merged_data[new_col] = merged_data[old_col]
            
            return merged_data
        else:
            # Fallback: create simple merged dataset
            logger.warning("Using fallback merge for business analysis")
            return pd.DataFrame()
    
    def _execute_visualization(self):
        """Execute comprehensive visualization generation."""
        logger.info("Executing visualization pipeline...")
        
        try:
            # Prepare data for visualization
            env_data = self.data['environmental_featured']
            business_data = self.data['business_featured']
            merged_data = self._prepare_merged_data_for_analysis()
            
            # Create comprehensive visualizations
            visualizations = self.viz_engine.create_comprehensive_report(
                env_data,
                business_data,
                merged_data,
                self.results.get('business_analysis', {})
            )
            
            # Save all visualizations
            saved_files = self.viz_engine.save_all_visualizations(visualizations)
            
            # Store visualization results
            self.results['visualizations'] = {
                'created_count': len(visualizations),
                'saved_files': saved_files,
                'visualization_types': list(visualizations.keys())
            }
            
            logger.info(f"Visualization completed: {len(visualizations)} visualizations created")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise
    
    def _generate_final_report(self):
        """Generate comprehensive final report and executive summary."""
        logger.info("Generating final comprehensive report...")
        
        try:
            # Compile executive summary
            executive_summary = self._create_executive_summary()
            
            # Create comprehensive report
            full_report = self._create_comprehensive_report()
            
            # Save reports
            with open('results/reports/executive_summary.json', 'w') as f:
                json.dump(executive_summary, f, indent=2, default=str)
            
            with open('results/reports/comprehensive_report.json', 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            
            # Create readable markdown report
            markdown_report = self._create_markdown_report(executive_summary, full_report)
            with open('results/reports/analysis_report.md', 'w') as f:
                f.write(markdown_report)
            
            # Store final results
            self.results['final_report'] = {
                'executive_summary': executive_summary,
                'comprehensive_report': full_report,
                'analysis_completion_time': datetime.now().isoformat()
            }
            
            logger.info("Final report generation completed")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    def _create_executive_summary(self) -> Dict:
        """Create executive summary of key findings."""
        summary = {
            'project_overview': {
                'project_name': self.config['project_name'],
                'analysis_date': self.analysis_timestamp.isoformat(),
                'version': self.config['analysis_version']
            },
            'data_overview': {
                'environmental_measurements': self.results['data_collection']['environmental_points'],
                'business_profiles': self.results['data_collection']['business_profiles'],
                'spatial_matches': self.results['spatial_analysis']['joined_pairs']
            },
            'key_findings': [],
            'business_recommendations': [],
            'technical_achievements': []
        }
        
        # Extract key findings from business analysis
        if 'business_analysis' in self.results:
            business_insights = self.results['business_analysis'].get('business_insights', {})
            
            if 'key_findings' in business_insights:
                summary['key_findings'] = business_insights['key_findings']
            
            if 'recommendations' in business_insights:
                summary['business_recommendations'] = business_insights['recommendations']
        
        # Technical achievements
        summary['technical_achievements'] = [
            f"Processed {self.results['data_collection']['environmental_points']} environmental measurements",
            f"Analyzed {self.results['data_collection']['business_profiles']} business profiles",
            f"Created {self.results['feature_engineering']['environmental_features_added'] + self.results['feature_engineering']['business_features_added']} engineered features",
            f"Identified {self.results['spatial_analysis']['clusters_identified']} distinct micro-climate zones",
            f"Generated {self.results['visualizations']['created_count']} professional visualizations"
        ]
        
        return summary
    
