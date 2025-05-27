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
    
    def create_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on geographic location."""
        df_featured = df.copy()
        
        # Distance from city center (using NYC as example)
        city_center_lat, city_center_lon = 40.7128, -74.0060
        
        df_featured['distance_from_center'] = np.sqrt(
            (df_featured['latitude'] - city_center_lat) ** 2 +
            (df_featured['longitude'] - city_center_lon) ** 2
        ) * 111000  # Convert to approximate meters
        
        # Grid-based location encoding
        lat_bins = pd.cut(df_featured['latitude'], bins=10, labels=False)
        lon_bins = pd.cut(df_featured['longitude'], bins=10, labels=False)
        df_featured['location_grid'] = lat_bins * 10 + lon_bins
        
        # Quadrant classification
        median_lat = df_featured['latitude'].median()
        median_lon = df_featured['longitude'].median()
        
        df_featured['quadrant'] = (
            (df_featured['latitude'] >= median_lat).astype(int) * 2 +
            (df_featured['longitude'] >= median_lon).astype(int)
        )
        
        logger.info("Geospatial features created")
        return df_featured
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        df_featured = df.copy()
        
        # Environmental interactions
        if 'temperature_celsius' in df.columns and 'humidity_percent' in df.columns:
            # Heat index approximation
            df_featured['heat_index'] = (
                df_featured['temperature_celsius'] + 
                df_featured['humidity_percent'] * 0.01 * df_featured['temperature_celsius']
            )
        
        if 'air_quality_index' in df.columns and 'wind_speed_ms' in df.columns:
            # Pollution dispersion factor
            df_featured['pollution_dispersion'] = (
                df_featured['air_quality_index'] / (df_featured['wind_speed_ms'] + 0.1)
            )
        
        # Business-environment interactions
        if 'environmental_comfort_index' in df.columns and 'business_success_score' in df.columns:
            df_featured['comfort_success_interaction'] = (
                df_featured['environmental_comfort_index'] * df_featured['business_success_score']
            )
        
        # Rating-reviews interaction
        if 'rating' in df.columns and 'review_count' in df.columns:
            df_featured['rating_volume_score'] = (
                df_featured['rating'] * np.log1p(df_featured['review_count'])
            )
        
        logger.info("Interaction features created")
        return df_featured
    
    def create_aggregated_features(self, df: pd.DataFrame, group_cols: list, agg_cols: list) -> pd.DataFrame:
        """Create aggregated features based on grouping variables."""
        df_featured = df.copy()
        
        for group_col in group_cols:
            if group_col not in df.columns:
                continue
                
            for agg_col in agg_cols:
                if agg_col not in df.columns:
                    continue
                
                # Calculate group statistics
                group_stats = df.groupby(group_col)[agg_col].agg(['mean', 'std', 'count'])
                
                # Merge back to original dataframe
                df_featured[f'{group_col}_{agg_col}_mean'] = df_featured[group_col].map(group_stats['mean'])
                df_featured[f'{group_col}_{agg_col}_std'] = df_featured[group_col].map(group_stats['std'])
                df_featured[f'{group_col}_{agg_col}_count'] = df_featured[group_col].map(group_stats['count'])
                
                # Calculate relative position within group
                df_featured[f'{group_col}_{agg_col}_relative'] = (
                    (df_featured[agg_col] - df_featured[f'{group_col}_{agg_col}_mean']) /
                    (df_featured[f'{group_col}_{agg_col}_std'] + 1e-8)
                )
        
        logger.info(f"Aggregated features created for {len(group_cols)} groups and {len(agg_cols)} variables")
        return df_featured
    def create_ranking_features(self, df: pd.DataFrame, rank_cols: list) -> pd.DataFrame:
        """Create ranking and percentile features."""
        df_featured = df.copy()
        
        for col in rank_cols:
            if col not in df.columns:
                continue
            
            # Rank (1 = highest value)
            df_featured[f'{col}_rank'] = df_featured[col].rank(ascending=False, method='dense')
            
            # Percentile rank (0-1, 1 = highest)
            df_featured[f'{col}_percentile'] = df_featured[col].rank(pct=True)
            
            # Quartile assignment
            df_featured[f'{col}_quartile'] = pd.qcut(
                df_featured[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop'
            )
            
            # Z-score normalization
            df_featured[f'{col}_zscore'] = (
                (df_featured[col] - df_featured[col].mean()) / df_featured[col].std()
            )
        
        logger.info(f"Ranking features created for {len(rank_cols)} variables")
        return df_featured
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: list, 
                                  encoding_type: str = 'onehot') -> pd.DataFrame:
        """Encode categorical variables for machine learning."""
        df_featured = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            if encoding_type == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df_featured[col], prefix=col, drop_first=True)
                df_featured = pd.concat([df_featured, dummies], axis=1)
                
            elif encoding_type == 'label':
                # Label encoding
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_featured[f'{col}_encoded'] = self.encoders[col].fit_transform(df_featured[col].astype(str))
                else:
                    df_featured[f'{col}_encoded'] = self.encoders[col].transform(df_featured[col].astype(str))
            
            elif encoding_type == 'target':
                # Target encoding (for supervised learning)
                if 'business_success_score' in df.columns:
                    target_means = df_featured.groupby(col)['business_success_score'].mean()
                    df_featured[f'{col}_target_encoded'] = df_featured[col].map(target_means)
        
        logger.info(f"Categorical encoding completed for {len(categorical_cols)} variables")
        return df_featured
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_cols: list, 
                                scaling_type: str = 'standard') -> pd.DataFrame:
        """Scale numerical features for machine learning."""
        df_featured = df.copy()
        
        for col in numerical_cols:
            if col not in df.columns:
                continue
            
            if scaling_type == 'standard':
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_featured[f'{col}_scaled'] = self.scalers[col].fit_transform(df_featured[[col]])
                else:
                    df_featured[f'{col}_scaled'] = self.scalers[col].transform(df_featured[[col]])
            
            elif scaling_type == 'minmax':
                if col not in self.scalers:
                    self.scalers[col] = MinMaxScaler()
                    df_featured[f'{col}_normalized'] = self.scalers[col].fit_transform(df_featured[[col]])
                else:
                    df_featured[f'{col}_normalized'] = self.scalers[col].transform(df_featured[[col]])
        
        logger.info(f"Numerical scaling completed for {len(numerical_cols)} variables")
        return df_featured
    
    def create_pca_features(self, df: pd.DataFrame, feature_cols: list, n_components: int = 3) -> pd.DataFrame:
        """Create PCA features for dimensionality reduction."""
        df_featured = df.copy()
        
        # Select only numerical columns that exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient features for PCA")
            return df_featured
        
        # Prepare data
        pca_data = df_featured[available_cols].fillna(0)
        
        # Standardize first
        scaler = StandardScaler()
        pca_data_scaled = scaler.fit_transform(pca_data)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(available_cols)))
        pca_features = pca.fit_transform(pca_data_scaled)
        
        # Add PCA features to dataframe
        for i in range(pca.n_components_):
            df_featured[f'pca_component_{i+1}'] = pca_features[:, i]
        
        # Store explained variance info
        self.feature_definitions['pca_explained_variance'] = pca.explained_variance_ratio_
        
        logger.info(f"PCA features created: {pca.n_components_} components explaining {pca.explained_variance_ratio_.sum():.3f} of variance")
        return df_featured
    

