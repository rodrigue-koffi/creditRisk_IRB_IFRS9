import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Ingénierie des features pour la modélisation
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.woe_mappings = {}
        
    def create_all_features(self) -> pd.DataFrame:
        """
        Créer toutes les features nécessaires
        """
        self._create_macro_features()
        self._create_behavioral_features()
        self._create_interaction_features()
        return self.df
    
    def _create_macro_features(self):
        """Créer des features macroéconomiques"""
        if 'UnemploymentRate' in self.df.columns:
            self.df['UnemploymentLag1'] = self.df['UnemploymentRate'].shift(1)
            self.df['UnemploymentChange'] = self.df['UnemploymentRate'].diff()
            self.df['GdpGrowthLag1'] = self.df['GdpGrowth'].shift(1)
    
    def _create_behavioral_features(self):
        """Créer des features comportementales"""
        self.df['AgeGroup'] = pd.cut(self.df['Age'], bins=[0, 25, 35, 50, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
        self.df['HighLoanAmount'] = (self.df['CreditAmount'] > self.df['CreditAmount'].median()).astype(int)
        self.df['LongDuration'] = (self.df['Duration'] > self.df['Duration'].median()).astype(int)
        
    def _create_interaction_features(self):
        """Créer des features d'interaction"""
        self.df['LoanPerYear'] = self.df['CreditAmount'] / self.df['Duration']
        self.df['RiskScore'] = (
            self.df['SavingAccountsNum'] * 0.3 +
            self.df['CheckingAccountNum'] * 0.4 +
            self.df['HousingNum'] * 0.2 +
            self.df['Job'] * 0.1
        )