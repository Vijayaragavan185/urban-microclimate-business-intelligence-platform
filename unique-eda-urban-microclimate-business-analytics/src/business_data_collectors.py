import requests
import json
from geopy.distance import geodesic
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
from data_collection import env_df
def create_simulated_business_data():
    """
    Create realistic business data around our environmental measurement points.
    
    Concepts:
    - Business success metrics
    - Geographic distribution
    - Realistic data relationships
    - Data quality considerations
    """
    
    print("🏪 Creating realistic business performance data...")
    
    # Business categories commonly found in urban areas
    business_categories = [
        'restaurant', 'cafe', 'retail_store', 'pharmacy', 
        'bank', 'gym', 'hair_salon', 'dry_cleaner'
    ]
    
    business_data = []
    business_id_counter = 1
    
    # Create businesses near each environmental measurement point
    for _, env_point in env_df.iterrows():
        # Create 3-7 businesses per environmental point
        num_businesses = np.random.randint(3, 8)
        
        for i in range(num_businesses):
            # Slightly offset business location from measurement point
            lat_offset = np.random.normal(0, 0.001)  # ~100m variation
            lon_offset = np.random.normal(0, 0.001)
            
            business_lat = env_point['latitude'] + lat_offset
            business_lon = env_point['longitude'] + lon_offset
            
            # Select business category
            category = np.random.choice(business_categories)
            
            # Generate realistic business metrics
            # Better environmental conditions = slightly better business performance
            env_bonus = 0
            if env_point['temperature_celsius'] > 18 and env_point['temperature_celsius'] < 25:
                env_bonus += 0.2  # Good temperature
            if env_point['air_quality_index'] < 75:
                env_bonus += 0.2  # Good air quality
            
            # Base rating with environmental influence
            base_rating = 3.5 + np.random.normal(0, 0.8) + env_bonus
            rating = max(1.0, min(5.0, base_rating))
            
            # Review count (more popular places have more reviews)
            review_count = max(1, int(np.random.exponential(50) * (rating / 3.5)))
            
            # Price level (1-4, with 2-3 being most common)
            price_level = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
            
            # Operating status (most businesses are open)
            is_open = np.random.choice([True, False], p=[0.85, 0.15])
            
            business = {
                'business_id': f'BIZ_{business_id_counter:04d}',
                'name': f'{category.title()} #{i+1} - {env_point["street_name"][:10]}',
                'category': category,
                'latitude': round(business_lat, 6),
                'longitude': round(business_lon, 6),
                'rating': round(rating, 1),
                'review_count': review_count,
                'price_level': price_level,
                'is_open': is_open,
                'measurement_area': env_point['street_name'],
                'collection_date': datetime.now()
            }
            
            business_data.append(business)
            business_id_counter += 1
    
    business_df = pd.DataFrame(business_data)
    print(f"✅ Created {len(business_df)} businesses across {len(env_df)} measurement areas")
    
    return business_df

# Create business dataset
business_df = create_simulated_business_data()
print("\n📊 Business Data Sample:")
print(business_df[['name', 'category', 'rating', 'review_count', 'is_open']].head())
