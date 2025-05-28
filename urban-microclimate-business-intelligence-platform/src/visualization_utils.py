
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class VisualizationEngine:
    
    def __init__(self):
        self.figure_counter = 0
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.setup_style()
        logger.info("Visualization engine initialized")
    
    def setup_style(self):
        """Configure professional plotting style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('viridis')
        
        # Professional matplotlib settings
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })


    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                 title: str = "Correlation Matrix") -> plt.Figure:
        """Create professional correlation heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            ax=ax
        )
        
        ax.set_title(title, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        logger.info(f"Correlation heatmap created: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
        return fig
    
    def create_environmental_dashboard(self, env_data: pd.DataFrame) -> plt.Figure:
        """Create comprehensive environmental conditions dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Environmental Conditions Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Temperature distribution
        if 'temperature_celsius' in env_data.columns:
            sns.histplot(data=env_data, x='temperature_celsius', kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Temperature Distribution')
            axes[0, 0].axvline(env_data['temperature_celsius'].mean(), color='red', 
                              linestyle='--', label=f"Mean: {env_data['temperature_celsius'].mean():.1f}°C")
            axes[0, 0].legend()
        
        # Humidity distribution
        if 'humidity_percent' in env_data.columns:
            sns.histplot(data=env_data, x='humidity_percent', kde=True, ax=axes[0, 1])
            axes[0, 1].set_title('Humidity Distribution')
            axes[0, 1].axvline(env_data['humidity_percent'].mean(), color='red', 
                              linestyle='--', label=f"Mean: {env_data['humidity_percent'].mean():.1f}%")
            axes[0, 1].legend()
        
        # Air Quality Index
        if 'air_quality_index' in env_data.columns:
            sns.histplot(data=env_data, x='air_quality_index', kde=True, ax=axes[0, 2])
            axes[0, 2].set_title('Air Quality Index Distribution')
            
            # Add AQI category lines
            aqi_lines = [50, 100, 150, 200]
            aqi_labels = ['Good', 'Moderate', 'Unhealthy', 'Very Unhealthy']
            colors = ['green', 'yellow', 'orange', 'red']
            
            for line, label, color in zip(aqi_lines, aqi_labels, colors):
                if line <= env_data['air_quality_index'].max():
                    axes[0, 2].axvline(line, color=color, linestyle=':', alpha=0.7, label=label)
            axes[0, 2].legend()
        
        # Comfort Index distribution
        if 'comfort_index' in env_data.columns:
            sns.histplot(data=env_data, x='comfort_index', kde=True, ax=axes[1, 0])
            axes[1, 0].set_title('Environmental Comfort Index')
            axes[1, 0].set_xlabel('Comfort Index (0-1)')
        
        # Wind Speed vs Temperature scatter
        if 'wind_speed_ms' in env_data.columns and 'temperature_celsius' in env_data.columns:
            scatter = axes[1, 1].scatter(env_data['temperature_celsius'], env_data['wind_speed_ms'],
                                       c=env_data.get('comfort_index', 'blue'), 
                                       cmap='viridis', alpha=0.7)
            axes[1, 1].set_title('Wind Speed vs Temperature')
            axes[1, 1].set_xlabel('Temperature (°C)')
            axes[1, 1].set_ylabel('Wind Speed (m/s)')
            
            if 'comfort_index' in env_data.columns:
                plt.colorbar(scatter, ax=axes[1, 1], label='Comfort Index')
        
        # Location-based analysis
        if 'street_name' in env_data.columns and 'comfort_index' in env_data.columns:
            location_comfort = env_data.groupby('street_name')['comfort_index'].mean().sort_values()
            
            bars = axes[1, 2].barh(range(len(location_comfort)), location_comfort.values)
            axes[1, 2].set_yticks(range(len(location_comfort)))
            axes[1, 2].set_yticklabels([name[:15] + '...' if len(name) > 15 else name 
                                       for name in location_comfort.index])
            axes[1, 2].set_title('Comfort Index by Location')
            axes[1, 2].set_xlabel('Average Comfort Index')
            
            # Color bars based on comfort level
            for i, bar in enumerate(bars):
                comfort = location_comfort.values[i]
                if comfort >= 0.7:
                    bar.set_color('green')
                elif comfort >= 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        plt.tight_layout()
        logger.info("Environmental dashboard created")
        return fig
