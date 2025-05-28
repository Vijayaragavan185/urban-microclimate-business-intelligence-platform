
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
    