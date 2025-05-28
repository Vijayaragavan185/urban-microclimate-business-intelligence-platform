
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

    
    def create_business_performance_dashboard(self, business_data: pd.DataFrame) -> plt.Figure:
        """Create comprehensive business performance dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Business Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Success Score distribution
        if 'business_success_score' in business_data.columns:
            sns.histplot(data=business_data, x='business_success_score', kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Business Success Score Distribution')
            axes[0, 0].axvline(business_data['business_success_score'].mean(), color='red', 
                              linestyle='--', label=f"Mean: {business_data['business_success_score'].mean():.3f}")
            axes[0, 0].legend()
        
        # Rating distribution by category
        if 'business_category' in business_data.columns and 'business_rating' in business_data.columns:
            sns.boxplot(data=business_data, x='business_category', y='business_rating', ax=axes[0, 1])
            axes[0, 1].set_title('Rating Distribution by Category')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Review count vs Rating scatter
        if 'business_reviews' in business_data.columns and 'business_rating' in business_data.columns:
            scatter = axes[0, 2].scatter(business_data['business_reviews'], business_data['business_rating'],
                                       c=business_data.get('business_success_score', 'blue'),
                                       cmap='viridis', alpha=0.7)
            axes[0, 2].set_title('Reviews vs Rating')
            axes[0, 2].set_xlabel('Review Count')
            axes[0, 2].set_ylabel('Rating')
            axes[0, 2].set_xscale('log')
            
            if 'business_success_score' in business_data.columns:
                plt.colorbar(scatter, ax=axes[0, 2], label='Success Score')
        
        # Success score by price level
        if 'business_price_level' in business_data.columns and 'business_success_score' in business_data.columns:
            price_success = business_data.groupby('business_price_level')['business_success_score'].mean()
            axes[1, 0].bar(price_success.index, price_success.values)
            axes[1, 0].set_title('Success Score by Price Level')
            axes[1, 0].set_xlabel('Price Level')
            axes[1, 0].set_ylabel('Average Success Score')
        
        # Category performance comparison
        if 'business_category' in business_data.columns and 'business_success_score' in business_data.columns:
            category_stats = business_data.groupby('business_category').agg({
                'business_success_score': ['mean', 'count']
            }).round(3)
            
            category_means = category_stats['business_success_score']['mean'].sort_values()
            bars = axes[1, 1].barh(range(len(category_means)), category_means.values)
            axes[1, 1].set_yticks(range(len(category_means)))
            axes[1, 1].set_yticklabels(category_means.index)
            axes[1, 1].set_title('Average Success Score by Category')
            
            # Add count annotations
            for i, (bar, count) in enumerate(zip(bars, category_stats['business_success_score']['count'][category_means.index])):
                axes[1, 1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'n={count}', va='center', fontsize=9)
        
        # Performance categories pie chart
        if 'business_success_score' in business_data.columns:
            # Create performance categories
            performance_categories = pd.cut(
                business_data['business_success_score'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['Poor', 'Average', 'Good', 'Excellent']
            )
            
            category_counts = performance_categories.value_counts()
            colors = ['red', 'orange', 'lightgreen', 'green']
            
            axes[1, 2].pie(category_counts.values, labels=category_counts.index, 
                          autopct='%1.1f%%', colors=colors)
            axes[1, 2].set_title('Business Performance Distribution')
        
        plt.tight_layout()
        logger.info("Business performance dashboard created")
        return fig
    
    def create_interactive_correlation_plot(self, merged_data: pd.DataFrame) -> go.Figure:
        """Create interactive correlation scatter plot."""
        if 'env_comfort_index' not in merged_data.columns or 'business_success_score' not in merged_data.columns:
            logger.warning("Required columns not found for correlation plot")
            return go.Figure()
        
        fig = go.Figure()
        
        # Color by business category if available
        if 'business_category' in merged_data.columns:
            categories = merged_data['business_category'].unique()
            colors = px.colors.qualitative.Set1[:len(categories)]
            
            for i, category in enumerate(categories):
                category_data = merged_data[merged_data['business_category'] == category]
                
                fig.add_trace(go.Scatter(
                    x=category_data['env_comfort_index'],
                    y=category_data['business_success_score'],
                    mode='markers',
                    name=category.title(),
                    marker=dict(
                        size=category_data.get('business_reviews', 10) / 5,
                        color=colors[i],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=
                    "<b>%{customdata[0]}</b><br>" +
                    "Environmental Comfort: %{x:.3f}<br>" +
                    "Business Success: %{y:.3f}<br>" +
                    "Rating: %{customdata[1]:.1f}<br>" +
                    "Reviews: %{customdata[2]}<br>" +
                    "<extra></extra>",
                    customdata=category_data[['business_name', 'business_rating', 'business_reviews']].values
                ))
        else:
            fig.add_trace(go.Scatter(
                x=merged_data['env_comfort_index'],
                y=merged_data['business_success_score'],
                mode='markers',
                marker=dict(
                    size=merged_data.get('business_reviews', 10) / 5,
                    color=merged_data.get('business_rating', 3.5),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Business Rating")
                ),
                hovertemplate=
                "Environmental Comfort: %{x:.3f}<br>" +
                "Business Success: %{y:.3f}<br>" +
                "<extra></extra>"
            ))
        
        # Add trendline
        if len(merged_data) > 2:
            z = np.polyfit(merged_data['env_comfort_index'], merged_data['business_success_score'], 1)
            p = np.poly1d(z)
            
            x_trend = np.linspace(merged_data['env_comfort_index'].min(),
                                merged_data['env_comfort_index'].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Trend Line',
                line=dict(dash='dash', color='red', width=2),
                hovertemplate="Trend Line<extra></extra>"
            ))
        
        fig.update_layout(
            title=dict(
                text='Environmental Comfort vs Business Success',
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title='Environmental Comfort Index',
            yaxis_title='Business Success Score',
            hovermode='closest',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=900,
            height=600
        )
        
        logger.info("Interactive correlation plot created")
        return fig

    
    def create_geospatial_map(self, merged_data: pd.DataFrame) -> folium.Map:
        """Create interactive geospatial map with business and environmental data."""
        if 'latitude' not in merged_data.columns or 'longitude' not in merged_data.columns:
            logger.warning("Geographic coordinates not found")
            return folium.Map()
        
        # Calculate map center
        center_lat = merged_data['latitude'].mean()
        center_lon = merged_data['longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add business markers
        for idx, row in merged_data.iterrows():
            # Determine marker color based on success score
            if 'business_success_score' in row:
                success = row['business_success_score']
                if success >= 0.7:
                    color = 'green'
                elif success >= 0.5:
                    color = 'orange'
                else:
                    color = 'red'
            else:
                color = 'blue'
            
            # Create popup text
            popup_text = f"""
            <b>{row.get('business_name', 'Business')}</b><br>
            Category: {row.get('business_category', 'Unknown')}<br>
            Rating: {row.get('business_rating', 'N/A')}<br>
            Success Score: {row.get('business_success_score', 'N/A'):.3f}<br>
            Environmental Comfort: {row.get('env_comfort_index', 'N/A'):.3f}<br>
            Temperature: {row.get('env_temperature', 'N/A')}°C<br>
            Air Quality: {row.get('env_air_quality', 'N/A')} AQI
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=row.get('business_name', f'Business {idx}'),
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
        
        # Add heatmap layer for environmental comfort
        if 'env_comfort_index' in merged_data.columns:
            heat_data = [
                [row['latitude'], row['longitude'], row['env_comfort_index']]
                for idx, row in merged_data.iterrows()
                if pd.notna(row['env_comfort_index'])
            ]
            
            if heat_data:
                HeatMap(
                    heat_data,
                    name='Environmental Comfort Heatmap',
                    radius=20,
                    blur=15,
                    max_zoom=1
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        logger.info(f"Geospatial map created with {len(merged_data)} points")
        return m
    
    def create_comprehensive_report(self, env_data: pd.DataFrame, business_data: pd.DataFrame, 
                                  merged_data: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Create comprehensive visualization report."""
        logger.info("Creating comprehensive visualization report")
        
        visualizations = {}
        
        try:
            # Environmental dashboard
            visualizations['environmental_dashboard'] = self.create_environmental_dashboard(env_data)
            
            # Business dashboard  
            visualizations['business_dashboard'] = self.create_business_performance_dashboard(business_data)
            
            # Correlation analysis
            if 'correlation_matrix' in analysis_results:
                visualizations['correlation_heatmap'] = self.create_correlation_heatmap(
                    analysis_results['correlation_matrix'],
                    "Environmental-Business Correlation Matrix"
                )
            
            # Interactive plots
            visualizations['interactive_correlation'] = self.create_interactive_correlation_plot(merged_data)
            
            # Geospatial map
            visualizations['geospatial_map'] = self.create_geospatial_map(merged_data)
            
            logger.info(f"Comprehensive report created with {len(visualizations)} visualizations")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return visualizations

    
    def save_all_visualizations(self, visualizations: Dict, output_dir: str = "results/visualizations"):
        """Save all visualizations to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        for name, viz in visualizations.items():
            try:
                if isinstance(viz, plt.Figure):
                    # Save matplotlib figures
                    filepath = f"{output_dir}/{name}.png"
                    viz.savefig(filepath, dpi=300, bbox_inches='tight')
                    saved_files.append(filepath)
                    
                elif hasattr(viz, 'write_html'):
                    # Save plotly figures
                    filepath = f"{output_dir}/{name}.html"
                    viz.write_html(filepath)
                    saved_files.append(filepath)
                    
                elif hasattr(viz, 'save'):
                    # Save folium maps
                    filepath = f"{output_dir}/{name}.html"
                    viz.save(filepath)
                    saved_files.append(filepath)
                    
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")
        
        logger.info(f"Saved {len(saved_files)} visualizations to {output_dir}")
        return saved_files


def test_visualization_engine():
    """Test visualization engine with sample data."""
    np.random.seed(42)
    
    # Sample environmental data
    env_data = pd.DataFrame({
        'latitude': 40.75 + np.random.normal(0, 0.01, 10),
        'longitude': -74.0 + np.random.normal(0, 0.01, 10),
        'street_name': [f'Street_{i}' for i in range(10)],
        'temperature_celsius': 20 + np.random.normal(0, 5, 10),
        'humidity_percent': 60 + np.random.normal(0, 15, 10),
        'air_quality_index': 50 + np.random.normal(0, 20, 10),
        'wind_speed_ms': 2 + np.random.exponential(1, 10),
        'comfort_index': np.random.uniform(0.3, 0.9, 10)
    })
    
    # Sample business data  
    business_data = pd.DataFrame({
        'business_name': [f'Business_{i}' for i in range(20)],
        'business_category': np.random.choice(['restaurant', 'cafe', 'retail'], 20),
        'business_rating': 3.5 + np.random.normal(0, 0.8, 20),
        'business_reviews': np.random.randint(10, 200, 20),
        'business_success_score': np.random.uniform(0.2, 0.9, 20),
        'business_price_level': np.random.randint(1, 5, 20)
    })
    
    # Sample merged data
    merged_data = pd.DataFrame({
        'latitude': 40.75 + np.random.normal(0, 0.01, 15),
        'longitude': -74.0 + np.random.normal(0, 0.01, 15),
        'business_name': [f'Business_{i}' for i in range(15)],
        'business_category': np.random.choice(['restaurant', 'cafe', 'retail'], 15),
        'business_rating': 3.5 + np.random.normal(0, 0.8, 15),
        'business_reviews': np.random.randint(10, 200, 15),
        'business_success_score': np.random.uniform(0.2, 0.9, 15),
        'env_comfort_index': np.random.uniform(0.3, 0.9, 15),
        'env_temperature': 20 + np.random.normal(0, 5, 15),
        'env_air_quality': 50 + np.random.normal(0, 20, 15)
    })
    
    # Test visualization engine
    viz_engine = VisualizationEngine()
    
    # Create sample analysis results
    analysis_results = {
        'correlation_matrix': merged_data[['business_success_score', 'env_comfort_index', 
                                         'business_rating', 'env_temperature']].corr()
    }
    
    # Create comprehensive report
    visualizations = viz_engine.create_comprehensive_report(
        env_data, business_data, merged_data, analysis_results
    )
    
    print(f"✅ Created {len(visualizations)} visualizations")
    return viz_engine, visualizations


if __name__ == "__main__":
    engine, vizs = test_visualization_engine()
    print("✅ Visualization engine module completed successfully!")
