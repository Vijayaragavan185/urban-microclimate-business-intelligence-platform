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
