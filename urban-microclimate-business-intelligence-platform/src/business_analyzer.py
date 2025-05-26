import pandas as pd
import numpy as np
from scipy import stats
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    
    def run_comprehensive_analysis(self) -> AnalysisResults:
        """Run complete analysis with generated sample data."""
        logger.info("Starting comprehensive business performance analysis")
        
        # Generate sample data for demonstration
        self._generate_sample_data()
        
        # Perform spatial merge
        self.spatial_merge_datasets()
        
        # Run all analyses
        correlation_matrix = self.calculate_correlation_matrix()
        statistical_tests = self.perform_statistical_significance_tests()
        cluster_analysis = self.perform_cluster_analysis()
        business_insights = self.generate_business_insights()
        
        # Compile results
        results = AnalysisResults(
            correlation_matrix=correlation_matrix,
            statistical_tests=statistical_tests,
            cluster_analysis=cluster_analysis,
            performance_metrics={},
            business_insights=business_insights
        )
        
        logger.info("Comprehensive analysis completed successfully")
        return results
    
    def _generate_sample_data(self):
        """Generate realistic sample data for analysis."""
        np.random.seed(42)
        
        # Generate environmental data
        locations = [
            {'lat': 40.7128, 'lon': -74.0060, 'name': 'Times Square'},
            {'lat': 40.7589, 'lon': -73.9851, 'name': 'Central Park'},
            {'lat': 40.7505, 'lon': -73.9934, 'name': 'Broadway'},
            {'lat': 40.7282, 'lon': -74.0776, 'name': 'Financial District'},
            {'lat': 40.7831, 'lon': -73.9712, 'name': 'Upper East Side'},
            {'lat': 40.7614, 'lon': -73.9776, 'name': 'Lincoln Center'},
            {'lat': 40.7549, 'lon': -73.9840, 'name': 'Columbus Circle'}
        ]
        
        env_data = []
        for i, loc in enumerate(locations):
            temp = 20 + np.random.normal(0, 5)
            humidity = 60 + np.random.normal(0, 15)
            aqi = 50 + np.random.normal(0, 20)
            wind = 2 + np.random.exponential(1)
            
            # Calculate comfort index
            temp_comfort = max(0, 1 - abs(temp - 20) / 15)
            humidity_comfort = max(0, 1 - abs(humidity - 50) / 40)
            aqi_comfort = max(0, 1 - aqi / 150)
            wind_comfort = max(0, 1 - abs(wind - 2) / 5)
            
            comfort_index = (temp_comfort * 0.3 + aqi_comfort * 0.3 + 
                           humidity_comfort * 0.2 + wind_comfort * 0.2)
            
            env_point = {
                'id': i + 1,
                'latitude': loc['lat'],
                'longitude': loc['lon'],
                'street_name': loc['name'],
                'temperature_celsius': round(temp, 1),
                'humidity_percent': round(np.clip(humidity, 0, 100), 1),
                'air_quality_index': int(np.clip(aqi, 0, 300)),
                'wind_speed_ms': round(max(0, wind), 1),
                'comfort_index': round(comfort_index, 3),
                'quality_score': np.random.uniform(0.7, 1.0)
            }
            env_data.append(env_point)
        
        self.env_data = pd.DataFrame(env_data)
        
        # Generate business data
        business_data = []
        business_id = 1
        
        for _, env_point in self.env_data.iterrows():
            n_businesses = np.random.randint(4, 8)
            
            for j in range(n_businesses):
                # Environmental influence
                env_bonus = (env_point['comfort_index'] - 0.5) * 0.4
                
                rating = 3.5 + env_bonus + np.random.normal(0, 0.5)
                rating = np.clip(rating, 1, 5)
                
                reviews = int(50 * (1 + env_bonus) * np.random.uniform(0.5, 2))
                success_score = (rating - 1) / 4 * 0.6 + np.log1p(reviews) / np.log1p(200) * 0.4
                
                business = {
                    'business_id': f'BIZ_{business_id:03d}',
                    'name': f'Business {business_id}',
                    'category': np.random.choice(['restaurant', 'cafe', 'retail', 'fitness']),
                    'latitude': env_point['latitude'] + np.random.normal(0, 0.001),
                    'longitude': env_point['longitude'] + np.random.normal(0, 0.001),
                    'rating': round(rating, 1),
                    'review_count': reviews,
                    'success_score': round(success_score, 3),
                    'price_level': np.random.randint(1, 5),
                    'is_open': np.random.choice([True, False], p=[0.9, 0.1])
                }
                
                business_data.append(business)
                business_id += 1
        
        self.business_data = pd.DataFrame(business_data)
        logger.info(f"Generated {len(self.env_data)} env points, {len(self.business_data)} businesses")
    
    def spatial_merge_datasets(self, max_distance: float = 200.0) -> pd.DataFrame:
        """Perform spatial join between environmental and business data."""
        if self.env_data is None or self.business_data is None:
            raise ValueError("Data not loaded. Call _generate_sample_data first.")
        
        logger.info(f"Performing spatial merge with max distance: {max_distance}m")
        
        merged_records = []
        
        for _, business in self.business_data.iterrows():
            distances = []
            for _, env_point in self.env_data.iterrows():
                # Simple distance calculation
                lat_diff = business['latitude'] - env_point['latitude']
                lon_diff = business['longitude'] - env_point['longitude']
                
                lat_meters = lat_diff * 111000
                lon_meters = lon_diff * 111000 * np.cos(np.radians(env_point['latitude']))
                distance = np.sqrt(lat_meters**2 + lon_meters**2)
                
                distances.append(distance)
            
            # Find closest environmental point
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            if min_distance <= max_distance:
                env_record = self.env_data.iloc[min_distance_idx]
                
                merged_record = {
                    'business_id': business['business_id'],
                    'business_name': business['name'],
                    'business_category': business['category'],
                    'business_rating': business['rating'],
                    'business_reviews': business['review_count'],
                    'business_success_score': business['success_score'],
                    'business_price_level': business['price_level'],
                    'business_is_open': business['is_open'],
                    'env_temperature': env_record['temperature_celsius'],
                    'env_humidity': env_record['humidity_percent'],
                    'env_air_quality': env_record['air_quality_index'],
                    'env_wind_speed': env_record['wind_speed_ms'],
                    'env_comfort_index': env_record['comfort_index'],
                    'env_quality_score': env_record['quality_score'],
                    'distance_to_env_point': min_distance,
                    'measurement_area': env_record['street_name'],
                    'latitude': business['latitude'],
                    'longitude': business['longitude']
                }
                
                merged_records.append(merged_record)
        
        self.merged_data = pd.DataFrame(merged_records)
        logger.info(f"Spatial merge completed: {len(self.merged_data)} business-environment pairs")
        
        return self.merged_data
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between environmental and business factors."""
        if self.merged_data is None:
            raise ValueError("Merged data not available.")
        
        correlation_columns = [
            'env_temperature', 'env_humidity', 'env_air_quality', 'env_wind_speed', 'env_comfort_index',
            'business_rating', 'business_reviews', 'business_success_score', 'business_price_level'
        ]
        
        available_columns = [col for col in correlation_columns if col in self.merged_data.columns]
        correlation_data = self.merged_data[available_columns].copy()
        correlation_data = correlation_data.dropna()
        
        correlation_matrix = correlation_data.corr()
        logger.info(f"Correlation matrix calculated for {len(available_columns)} variables")
        return correlation_matrix
    
    def perform_statistical_significance_tests(self) -> Dict:
        """Perform statistical tests for key relationships."""
        if self.merged_data is None:
            raise ValueError("Merged data not available.")
        
        test_results = {}
        
        # Test: Environmental comfort vs Business success
        if 'env_comfort_index' in self.merged_data.columns and 'business_success_score' in self.merged_data.columns:
            aligned_data = self.merged_data[['env_comfort_index', 'business_success_score']].dropna()
            
            if len(aligned_data) > 5:
                correlation, p_value = stats.pearsonr(
                    aligned_data['env_comfort_index'], 
                    aligned_data['business_success_score']
                )
                
                test_results['comfort_vs_success'] = {
                    'correlation': round(correlation, 4),
                    'p_value': round(p_value, 4),
                    'significant': p_value < 0.05,
                    'sample_size': len(aligned_data),
                    'interpretation': self._interpret_correlation(correlation, p_value)
                }
        
        # Test: Air quality vs Business rating
        if 'env_air_quality' in self.merged_data.columns and 'business_rating' in self.merged_data.columns:
            aligned_data = self.merged_data[['env_air_quality', 'business_rating']].dropna()
            
            if len(aligned_data) > 5:
                correlation, p_value = stats.pearsonr(
                    aligned_data['env_air_quality'], 
                    aligned_data['business_rating']
                )
                
                test_results['air_quality_vs_rating'] = {
                    'correlation': round(correlation, 4),
                    'p_value': round(p_value, 4),
                    'significant': p_value < 0.05,
                    'sample_size': len(aligned_data),
                    'interpretation': self._interpret_correlation(correlation, p_value)
                }
        
        logger.info(f"Statistical tests completed: {len(test_results)} tests performed")
        return test_results
    
    def _interpret_correlation(self, correlation: float, p_value: float) -> str:
        """Interpret correlation strength and significance."""
        if p_value >= 0.05:
            return "Not statistically significant"
        
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            strength = "Strong"
        elif abs_corr >= 0.5:
            strength = "Moderate"
        elif abs_corr >= 0.3:
            strength = "Weak"
        else:
            strength = "Very weak"
        
        direction = "positive" if correlation > 0 else "negative"
        return f"{strength} {direction} correlation (statistically significant)"
    
    def perform_cluster_analysis(self) -> Dict:
        """Perform basic clustering analysis."""
        if self.merged_data is None:
            raise ValueError("Merged data not available.")
        
        # Simple analysis by comfort categories
        if 'env_comfort_index' in self.merged_data.columns:
            comfort_categories = pd.cut(
                self.merged_data['env_comfort_index'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['Poor', 'Fair', 'Good', 'Excellent'],
                include_lowest=True
            )
            
            category_analysis = {}
            for category in comfort_categories.unique():
                if pd.notna(category):
                    category_data = self.merged_data[comfort_categories == category]
                    
                    category_analysis[str(category)] = {
                        'count': len(category_data),
                        'avg_business_success': category_data['business_success_score'].mean(),
                        'avg_business_rating': category_data['business_rating'].mean()
                    }
            
            cluster_analysis = {
                'method': 'comfort_categorization',
                'categories': category_analysis,
                'total_points': len(self.merged_data)
            }
            
            logger.info("Cluster analysis completed using comfort categories")
            return cluster_analysis
        
        return {}
    
    def generate_business_insights(self) -> Dict:
        """Generate actionable business insights."""
        if self.merged_data is None:
            raise ValueError("Merged data not available.")
        
        insights = {
            'summary_statistics': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Summary statistics
        insights['summary_statistics']['total_businesses'] = len(self.merged_data)
        insights['summary_statistics']['avg_success_score'] = round(self.merged_data['business_success_score'].mean(), 3)
        insights['summary_statistics']['avg_comfort_index'] = round(self.merged_data['env_comfort_index'].mean(), 3)
        
        # High vs low performers
        high_performers = self.merged_data[self.merged_data['business_success_score'] > 0.7]
        low_performers = self.merged_data[self.merged_data['business_success_score'] < 0.3]
        
        if len(high_performers) > 0:
            avg_comfort_high = high_performers['env_comfort_index'].mean()
            insights['key_findings'].append(
                f"High-performing businesses operate in areas with avg comfort index of {avg_comfort_high:.3f}"
            )
        
        if len(low_performers) > 0:
            avg_comfort_low = low_performers['env_comfort_index'].mean()
            insights['key_findings'].append(
                f"Low-performing businesses operate in areas with avg comfort index of {avg_comfort_low:.3f}"
            )
        
        # Correlation insights
        if 'env_comfort_index' in self.merged_data.columns and 'business_success_score' in self.merged_data.columns:
            comfort_success_corr = self.merged_data['env_comfort_index'].corr(self.merged_data['business_success_score'])
            
            if comfort_success_corr > 0.3:
                insights['recommendations'].append(
                    "Prioritize locations with high environmental comfort for new ventures"
                )
            elif comfort_success_corr < -0.3:
                insights['recommendations'].append(
                    "Investigate negative correlation between comfort and success"
                )
        
        logger.info("Business insights generation completed")
        return insights


def main_analysis_pipeline():
    """Execute the complete business analysis pipeline."""
    analyzer = BusinessPerformanceAnalyzer()
    
    try:
        results = analyzer.run_comprehensive_analysis()
        
        print("\\n=== BUSINESS PERFORMANCE ANALYSIS RESULTS ===")
        print(f"Merged dataset size: {len(analyzer.merged_data)} business-environment pairs")
        
        # Statistical tests
        if results.statistical_tests:
            print("\\n--- Statistical Significance Tests ---")
            for test_name, test_result in results.statistical_tests.items():
                print(f"{test_name}: {test_result['interpretation']}")
        
        # Cluster analysis
        if results.cluster_analysis:
            print(f"\\n--- Cluster Analysis ---")
            if 'categories' in results.cluster_analysis:
                for category, stats in results.cluster_analysis['categories'].items():
                    print(f"{category}: {stats['count']} businesses, avg success: {stats['avg_business_success']:.3f}")
        
        # Business insights
        if results.business_insights:
            print("\\n--- Key Business Insights ---")
            for finding in results.business_insights.get('key_findings', []):
                print(f"• {finding}")
            
            print("\\n--- Recommendations ---")
            for rec in results.business_insights.get('recommendations', []):
                print(f"• {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {e}")
        raise


if __name__ == "__main__":
    results = main_analysis_pipeline()
    print("\\n✅ Business analysis module completed successfully!")