import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_definitions = {}
        logger.info("Feature engineer initialized")