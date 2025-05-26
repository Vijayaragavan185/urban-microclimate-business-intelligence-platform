
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

    
    def create_buffer_zones(self, points_df: pd.DataFrame, buffer_radius: float = 200) -> pd.DataFrame:
        """Create buffer zones around points for analysis."""
        buffer_zones = []
        
        for idx, point in points_df.iterrows():
            # Create circular buffer (simplified as bounding box)
            lat_offset = buffer_radius / 111000  # Approximate degrees per meter
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
    
    def calculate_spatial_density(self, points_df: pd.DataFrame, grid_size: float = 500) -> pd.DataFrame:
        """Calculate point density across a spatial grid."""
        # Find bounds of all points
        min_lat, max_lat = points_df['latitude'].min(), points_df['latitude'].max()
        min_lon, max_lon = points_df['longitude'].min(), points_df['longitude'].max()
        
        # Create grid
        lat_step = grid_size / 111000  # Convert meters to degrees
        lon_step = grid_size / 111000  # Simplified (should account for latitude)
        
        lat_range = np.arange(min_lat, max_lat + lat_step, lat_step)
        lon_range = np.arange(min_lon, max_lon + lon_step, lon_step)
        
        density_grid = []
        
        for i, lat in enumerate(lat_range[:-1]):
            for j, lon in enumerate(lon_range[:-1]):
                # Count points in this grid cell
                points_in_cell = points_df[
                    (points_df['latitude'] >= lat) & 
                    (points_df['latitude'] < lat_range[i + 1]) &
                    (points_df['longitude'] >= lon) & 
                    (points_df['longitude'] < lon_range[j + 1])
                ]
                
                density_cell = {
                    'grid_i': i,
                    'grid_j': j,
                    'lat_min': lat,
                    'lat_max': lat_range[i + 1],
                    'lon_min': lon,
                    'lon_max': lon_range[j + 1],
                    'center_lat': (lat + lat_range[i + 1]) / 2,
                    'center_lon': (lon + lon_range[j + 1]) / 2,
                    'point_count': len(points_in_cell),
                    'density_per_sqkm': len(points_in_cell) / ((grid_size / 1000) ** 2)
                }
                
                density_grid.append(density_cell)
        
        return pd.DataFrame(density_grid)
    
    def create_spatial_clusters(self, points_df: pd.DataFrame, cluster_distance: float = 300) -> pd.DataFrame:
        """Group nearby points into spatial clusters."""
        points_with_clusters = points_df.copy()
        points_with_clusters['cluster_id'] = -1  # Unassigned
        
        cluster_id = 0
        unassigned_mask = points_with_clusters['cluster_id'] == -1
        
        while unassigned_mask.any():
            # Start new cluster with first unassigned point
            seed_idx = points_with_clusters[unassigned_mask].index[0]
            seed_point = points_with_clusters.loc[seed_idx]
            
            # Assign seed to new cluster
            points_with_clusters.loc[seed_idx, 'cluster_id'] = cluster_id
            cluster_points = [seed_idx]
            
            # Find all points within cluster_distance of any cluster point
            changed = True
            while changed:
                changed = False
                
                for cluster_point_idx in cluster_points:
                    cluster_point = points_with_clusters.loc[cluster_point_idx]
                    
                    # Find unassigned points near this cluster point
                    for idx, point in points_with_clusters[unassigned_mask].iterrows():
                        distance = geodesic(
                            (cluster_point['latitude'], cluster_point['longitude']),
                            (point['latitude'], point['longitude'])
                        ).meters
                        
                        if distance <= cluster_distance:
                            points_with_clusters.loc[idx, 'cluster_id'] = cluster_id
                            cluster_points.append(idx)
                            changed = True
                
                # Update unassigned mask
                unassigned_mask = points_with_clusters['cluster_id'] == -1
            
            cluster_id += 1
        
        # Add cluster summary statistics
        cluster_summary = points_with_clusters.groupby('cluster_id').agg({
            'latitude': ['count', 'mean', 'std'],
            'longitude': ['mean', 'std']
        }).round(6)
        
        logger.info(f"Created {cluster_id} spatial clusters")
        
        return points_with_clusters, cluster_summary
    
    def create_interactive_map(self, points_df: pd.DataFrame, 
                             center_lat: float = None, center_lon: float = None,
                             zoom_start: int = 12) -> folium.Map:
        """Create interactive map with points."""
        
        if center_lat is None:
            center_lat = points_df['latitude'].mean()
        if center_lon is None:
            center_lon = points_df['longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add points to map
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
    
    def create_heatmap(self, points_df: pd.DataFrame, weight_column: str = None) -> folium.Map:
        """Create density heatmap of points."""
        center_lat = points_df['latitude'].mean()
        center_lon = points_df['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12
        )
        
        # Prepare data for heatmap
        if weight_column and weight_column in points_df.columns:
            heat_data = [[row['latitude'], row['longitude'], row[weight_column]] 
                        for idx, row in points_df.iterrows()]
        else:
            heat_data = [[row['latitude'], row['longitude']] 
                        for idx, row in points_df.iterrows()]
        
        # Add heatmap
        HeatMap(heat_data).add_to(m)
        
        return m

