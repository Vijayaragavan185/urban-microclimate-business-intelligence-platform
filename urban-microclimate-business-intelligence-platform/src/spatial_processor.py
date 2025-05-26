
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import folium
from folium.plugins import HeatMap, MarkerCluster
import logging

logger = logging.getLogger(__name__)

class SpatialProcessor:
    
    def __init__(self):
        self.distance_matrix = None
        self.spatial_index = None
        logger.info("Spatial processor initialized")
    
    def calculate_distance_matrix(self, locations_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance matrix between all location pairs."""
        coords = locations_df[['latitude', 'longitude']].values
        
        # Calculate pairwise distances using geodesic (accurate for Earth)
        n_locations = len(coords)
        distance_matrix = np.zeros((n_locations, n_locations))
        
        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    distance_matrix[i, j] = geodesic(coords[i], coords[j]).meters
        
        self.distance_matrix = pd.DataFrame(
            distance_matrix, 
            index=locations_df.index, 
            columns=locations_df.index
        )
        
        logger.info(f"Distance matrix calculated for {n_locations} locations")
        return self.distance_matrix

    
    def find_nearest_neighbors(self, source_df: pd.DataFrame, target_df: pd.DataFrame, 
                              k: int = 5, max_distance: float = 500) -> pd.DataFrame:
        """Find k nearest neighbors for each source point in target dataset."""
        source_coords = source_df[['latitude', 'longitude']].values
        target_coords = target_df[['latitude', 'longitude']].values
        
        # Use sklearn NearestNeighbors for efficiency
        nbrs = NearestNeighbors(n_neighbors=min(k, len(target_coords)), metric='haversine')
        nbrs.fit(np.radians(target_coords))
        
        # Convert to radians for haversine distance
        distances, indices = nbrs.kneighbors(np.radians(source_coords))
        
        # Convert distances back to meters (haversine returns in radians)
        distances_meters = distances * 6371000  # Earth radius in meters
        
        neighbor_results = []
        
        for i, (source_idx, source_row) in enumerate(source_df.iterrows()):
            for j in range(len(indices[i])):
                distance_m = distances_meters[i, j]
                
                if distance_m <= max_distance:
                    target_idx = target_df.index[indices[i, j]]
                    target_row = target_df.loc[target_idx]
                    
                    neighbor_results.append({
                        'source_id': source_idx,
                        'target_id': target_idx,
                        'distance_meters': round(distance_m, 1),
                        'neighbor_rank': j + 1,
                        'source_lat': source_row['latitude'],
                        'source_lon': source_row['longitude'],
                        'target_lat': target_row['latitude'],
                        'target_lon': target_row['longitude']
                    })
        
        return pd.DataFrame(neighbor_results)
    
    def spatial_join(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                    max_distance: float = 200, join_type: str = 'nearest') -> pd.DataFrame:
        """Perform spatial join between two datasets."""
        
        if join_type == 'nearest':
            # Join each left point to nearest right point within max_distance
            joined_records = []
            
            for left_idx, left_row in left_df.iterrows():
                min_distance = float('inf')
                nearest_right_idx = None
                
                for right_idx, right_row in right_df.iterrows():
                    distance = geodesic(
                        (left_row['latitude'], left_row['longitude']),
                        (right_row['latitude'], right_row['longitude'])
                    ).meters
                    
                    if distance < min_distance and distance <= max_distance:
                        min_distance = distance
                        nearest_right_idx = right_idx
                
                if nearest_right_idx is not None:
                    # Combine left and right records
                    joined_record = {}
                    
                    # Add left data with prefix
                    for col in left_df.columns:
                        joined_record[f'left_{col}'] = left_row[col]
                    
                    # Add right data with prefix
                    right_row_data = right_df.loc[nearest_right_idx]
                    for col in right_df.columns:
                        joined_record[f'right_{col}'] = right_row_data[col]
                    
                    # Add spatial relationship info
                    joined_record['spatial_distance_meters'] = round(min_distance, 1)
                    joined_record['join_type'] = 'nearest_neighbor'
                    
                    joined_records.append(joined_record)
            
            return pd.DataFrame(joined_records)
        
        elif join_type == 'within_distance':
            # Join all right points within max_distance of each left point
            joined_records = []
            
            for left_idx, left_row in left_df.iterrows():
                matches_found = False
                
                for right_idx, right_row in right_df.iterrows():
                    distance = geodesic(
                        (left_row['latitude'], left_row['longitude']),
                        (right_row['latitude'], right_row['longitude'])
                    ).meters
                    
                    if distance <= max_distance:
                        matches_found = True
                        
                        # Combine records
                        joined_record = {}
                        
                        for col in left_df.columns:
                            joined_record[f'left_{col}'] = left_row[col]
                        
                        for col in right_df.columns:
                            joined_record[f'right_{col}'] = right_row[col]
                        
                        joined_record['spatial_distance_meters'] = round(distance, 1)
                        joined_record['join_type'] = 'within_distance'
                        
                        joined_records.append(joined_record)
                
                # If no matches found, include left record with nulls for right
                if not matches_found:
                    joined_record = {}
                    
                    for col in left_df.columns:
                        joined_record[f'left_{col}'] = left_row[col]
                    
                    for col in right_df.columns:
                        joined_record[f'right_{col}'] = None
                    
                    joined_record['spatial_distance_meters'] = None
                    joined_record['join_type'] = 'no_match'
                    
                    joined_records.append(joined_record)
            
            return pd.DataFrame(joined_records)
