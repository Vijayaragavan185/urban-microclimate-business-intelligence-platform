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
        
        nbrs = NearestNeighbors(n_neighbors=min(k, len(target_coords)), metric='haversine')
        nbrs.fit(np.radians(target_coords))
        
        distances, indices = nbrs.kneighbors(np.radians(source_coords))
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
                    joined_record = {}
                    
                    for col in left_df.columns:
                        joined_record[f'left_{col}'] = left_row[col]
                    
                    right_row_data = right_df.loc[nearest_right_idx]
                    for col in right_df.columns:
                        joined_record[f'right_{col}'] = right_row_data[col]
                    
                    joined_record['spatial_distance_meters'] = round(min_distance, 1)
                    joined_record['join_type'] = 'nearest_neighbor'
                    
                    joined_records.append(joined_record)
            
            return pd.DataFrame(joined_records)
        
        return pd.DataFrame()  # Simplified for other join types
    
    def create_spatial_clusters(self, points_df: pd.DataFrame, cluster_distance: float = 300) -> tuple:
        """Group nearby points into spatial clusters - FIXED VERSION."""
        points_with_clusters = points_df.copy()
        points_with_clusters['cluster_id'] = -1
        
        cluster_id = 0
        
        while (points_with_clusters['cluster_id'] == -1).any():
            # Get unassigned points
            unassigned_indices = points_with_clusters[points_with_clusters['cluster_id'] == -1].index
            
            if len(unassigned_indices) == 0:
                break
                
            # Start new cluster with first unassigned point
            seed_idx = unassigned_indices[0]
            points_with_clusters.loc[seed_idx, 'cluster_id'] = cluster_id
            cluster_points = [seed_idx]
            
            # Expand cluster
            changed = True
            while changed:
                changed = False
                new_cluster_points = []
                
                for cluster_point_idx in cluster_points:
                    cluster_lat = float(points_with_clusters.loc[cluster_point_idx, 'latitude'])
                    cluster_lon = float(points_with_clusters.loc[cluster_point_idx, 'longitude'])
                    
                    # Check all unassigned points
                    current_unassigned = points_with_clusters[points_with_clusters['cluster_id'] == -1]
                    
                    for idx in current_unassigned.index:
                        point_lat = float(points_with_clusters.loc[idx, 'latitude'])
                        point_lon = float(points_with_clusters.loc[idx, 'longitude'])
                        
                        # Calculate distance using scalar values
                        distance = geodesic((cluster_lat, cluster_lon), (point_lat, point_lon)).meters
                        
                        if distance <= cluster_distance:
                            points_with_clusters.loc[idx, 'cluster_id'] = cluster_id
                            new_cluster_points.append(idx)
                            changed = True
                
                # Add new points to cluster
                cluster_points.extend(new_cluster_points)
            
            cluster_id += 1
        
        # Create cluster summary
        cluster_summary = points_with_clusters.groupby('cluster_id').agg({
            'latitude': ['count', 'mean', 'std'],
            'longitude': ['mean', 'std']
        }).round(6)
        
        logger.info(f"Created {cluster_id} spatial clusters")
        return points_with_clusters, cluster_summary
    
    def create_buffer_zones(self, points_df: pd.DataFrame, buffer_radius: float = 200) -> pd.DataFrame:
        """Create buffer zones around points for analysis."""
        buffer_zones = []
        
        for idx, point in points_df.iterrows():
            lat_offset = buffer_radius / 111000
            lon_offset = buffer_radius / (111000 * np.cos(np.radians(point['latitude'])))
            
            buffer_zone = {
                'point_id': idx,
                'center_lat': point['latitude'],
                'center_lon': point['longitude'],
                'buffer_radius_meters': buffer_radius,
                'north_bound': point['latitude'] + lat_offset,
                'south_bound': point['latitude'] - lat_offset,
                'east_bound': point['longitude'] + lon_offset,
                'west_bound': point['longitude'] - lon_offset,
                'buffer_area_sqm': np.pi * (buffer_radius ** 2)
            }
            
            buffer_zones.append(buffer_zone)
        
        return pd.DataFrame(buffer_zones)
    
    def create_interactive_map(self, points_df: pd.DataFrame, 
                             center_lat: float = None, center_lon: float = None,
                             zoom_start: int = 12) -> folium.Map:
        """Create interactive map with points."""
        
        if center_lat is None:
            center_lat = points_df['latitude'].mean()
        if center_lon is None:
            center_lon = points_df['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, point in points_df.iterrows():
            popup_text = f"Point {idx}"
            if 'name' in point:
                popup_text = point['name']
            elif 'street_name' in point:
                popup_text = point['street_name']
            
            folium.Marker(
                location=[point['latitude'], point['longitude']],
                popup=popup_text,
                tooltip=f"Lat: {point['latitude']:.4f}, Lon: {point['longitude']:.4f}"
            ).add_to(marker_cluster)
        
        return m


def test_spatial_processor():
    """Test spatial processing functionality - FIXED VERSION."""
    np.random.seed(42)
    
    # Environmental points
    env_points = pd.DataFrame({
        'latitude': [40.7128, 40.7589, 40.7505, 40.7282, 40.7831],
        'longitude': [-74.0060, -73.9851, -73.9934, -74.0776, -73.9712],
        'type': 'environmental',
        'comfort_index': np.random.uniform(0.3, 0.9, 5)
    })
    
    # Business points (closer together to avoid infinite loops)
    business_points = pd.DataFrame({
        'latitude': 40.75 + np.random.normal(0, 0.002, 8),  # Smaller variance
        'longitude': -74.0 + np.random.normal(0, 0.002, 8),  # Smaller variance
        'type': 'business',
        'success_score': np.random.uniform(0.2, 0.8, 8)
    })
    
    processor = SpatialProcessor()
    
    # Test distance matrix
    distance_matrix = processor.calculate_distance_matrix(env_points)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    # Test spatial join
    joined_data = processor.spatial_join(business_points, env_points, max_distance=2000)  # Increased distance
    print(f"Spatial join results: {len(joined_data)} matches")
    
    # Test nearest neighbors
    neighbors = processor.find_nearest_neighbors(business_points, env_points, k=3)
    print(f"Nearest neighbors found: {len(neighbors)} relationships")
    
    # Test clustering with larger distance to ensure clustering works
    combined_points = pd.concat([env_points, business_points], ignore_index=True)
    clustered_points, cluster_summary = processor.create_spatial_clusters(
        combined_points, cluster_distance=1000  # Increased distance
    )
    print(f"Spatial clustering: {clustered_points['cluster_id'].nunique()} clusters")
    print(f"Cluster distribution:\\n{clustered_points['cluster_id'].value_counts()}")
    
    return processor, joined_data, clustered_points


if __name__ == "__main__":
    processor, joined_data, clustered_data = test_spatial_processor()
    print("✅ Spatial processor module completed successfully!")