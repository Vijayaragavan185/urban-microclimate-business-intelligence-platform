# Create project folders
import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Create professional folder structure
folders = [
    'unique-eda-urban-microclimate-business-analytics',
    'unique-eda-urban-microclimate-business-analytics/data',
    'unique-eda-urban-microclimate-business-analytics/data/raw',
    'unique-eda-urban-microclimate-business-analytics/data/processed',
    'unique-eda-urban-microclimate-business-analytics/notebooks',
    'unique-eda-urban-microclimate-business-analytics/src',
    'unique-eda-urban-microclimate-business-analytics/results'
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✅ Project structure created!")
print("This organized structure shows professionalism to recruiters")

def create_database():
    """
    Create a professional database schema for our analysis.
    
    Key Concepts:
    - Primary Keys: Unique identifier for each record
    - Foreign Keys: Connect related tables
    - Indexes: Speed up location-based queries
    - Data Types: Optimize storage and performance
    """
    
    # Connect to SQLite database (creates file if doesn't exist)
    conn = sqlite3.connect('unique-eda-urban-microclimate-business-analytics/data/microclimate.db')
    cursor = conn.cursor()
    
    # Environmental measurements table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS environmental_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        street_name TEXT,
        temperature_celsius REAL,
        humidity_percent REAL,
        air_quality_index INTEGER,
        noise_level_db REAL,
        wind_speed_ms REAL,
        comfort_index REAL,
        data_source TEXT
    )
    ''')
    
    # Business performance table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS business_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        business_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        category TEXT,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        rating REAL,
        review_count INTEGER,
        price_level INTEGER,
        is_open BOOLEAN,
        success_score REAL,
        collection_date DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create indexes for fast geospatial queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_env_location ON environmental_data(latitude, longitude)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_business_location ON business_data(latitude, longitude)')
    
    conn.commit()
    print("✅ Database created with professional schema!")
    print("📍 Geospatial indexes created for fast location queries")
    return conn

# Create our database
db = create_database()
