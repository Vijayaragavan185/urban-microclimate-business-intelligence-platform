
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
