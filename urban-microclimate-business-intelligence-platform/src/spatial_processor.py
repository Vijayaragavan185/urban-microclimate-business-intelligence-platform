
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
    
