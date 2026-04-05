import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import logging

logger = logging.getLogger(__name__)

class ScenarioGenerator:
    """
    Générateur de scénarios pour stress test
    """
    
    def __init__(self, macro_df: pd.DataFrame):
        self.macro_df = macro_df
        self.var_model = None
        
    def generate_regulatory_scenarios(self, horizon: int = 9) -> dict:
        """
        Générer les scénarios réglementaires (type CCAR/ECB)
        """
        # Scénario de base
        baseline = self._generate_baseline_scenario(horizon)
        
        # Scénario adverse
        adverse = self._generate_adverse_scenario(horizon)
        
        # Scénario très adverse
        severely_adverse = self._generate_severely_adverse_scenario(horizon)
        
        return {
            'Baseline': {'projections': baseline, 'weight': 0.70},
            'Adverse': {'projections': adverse, 'weight': 0.20},
            'SeverelyAdverse': {'projections': severely_adverse, 'weight': 0.10}
        }
    
    def _generate_baseline_scenario(self, horizon: int) -> pd.DataFrame:
        """
        Générer un scénario de base
        """
        last_values = {
            'UnemploymentRate': self.macro_df['UnemploymentRate'].iloc[-1],
            'GdpGrowth': self.macro_df['GdpGrowth'].iloc[-1],
            'InterestRate': self.macro_df['InterestRate'].iloc[-1]
        }
        
        projections = []
        for t in range(1, horizon + 1):
            projected = {
                'Quarter': t,
                'UnemploymentRate': last_values['UnemploymentRate'] * (1 - 0.02 * t),
                'GdpGrowth': last_values['GdpGrowth'] * (1 + 0.01 * t),
                'InterestRate': last_values['InterestRate'] * (1 + 0.005 * t)
            }
            projections.append(projected)
        
        return pd.DataFrame(projections)
    
    def _generate_adverse_scenario(self, horizon: int) -> pd.DataFrame:
        """
        Générer un scénario adverse
        """
        last_values = {
            'UnemploymentRate': self.macro_df['UnemploymentRate'].iloc[-1],
            'GdpGrowth': self.macro_df['GdpGrowth'].iloc[-1],
            'InterestRate': self.macro_df['InterestRate'].iloc[-1]
        }
        
        projections = []
        for t in range(1, horizon + 1):
            projected = {
                'Quarter': t,
                'UnemploymentRate': last_values['UnemploymentRate'] * (1 + 0.05 * t),
                'GdpGrowth': last_values['GdpGrowth'] * (1 - 0.02 * t),
                'InterestRate': last_values['InterestRate'] * (1 + 0.01 * t)
            }
            projections.append(projected)
        
        return pd.DataFrame(projections)
    
    def _generate_severely_adverse_scenario(self, horizon: int) -> pd.DataFrame:
        """
        Générer un scénario très adverse
        """
        last_values = {
            'UnemploymentRate': self.macro_df['UnemploymentRate'].iloc[-1],
            'GdpGrowth': self.macro_df['GdpGrowth'].iloc[-1],
            'InterestRate': self.macro_df['InterestRate'].iloc[-1]
        }
        
        projections = []
        for t in range(1, horizon + 1):
            projected = {
                'Quarter': t,
                'UnemploymentRate': last_values['UnemploymentRate'] * (1 + 0.10 * t),
                'GdpGrowth': last_values['GdpGrowth'] * (1 - 0.05 * t),
                'InterestRate': last_values['InterestRate'] * (1 + 0.02 * t)
            }
            projections.append(projected)
        
        return pd.DataFrame(projections)