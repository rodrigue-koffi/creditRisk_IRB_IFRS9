import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import logging

logger = logging.getLogger(__name__)

class MacroDataGenerator:
    """
    Générateur de données macroéconomiques et de scénarios
    """
    
    def __init__(self, macro_df: pd.DataFrame):
        self.macro_df = macro_df
        self.var_model = None
        
    def generate_scenarios(self, horizon: int = 5) -> dict:
        """
        Générer les 3 scénarios IFRS 9
        """
        scenarios = {
            'Baseline': {'weight': 0.50, 'shock': 0.0, 'description': 'Scénario central'},
            'Upside': {'weight': 0.25, 'shock': 0.5, 'description': 'Scénario haussier'},
            'Downside': {'weight': 0.25, 'shock': -0.5, 'description': 'Scénario baissier'}
        }
        
        scenario_projections = {}
        last_values = {
            'UnemploymentRate': self.macro_df['UnemploymentRate'].iloc[-1],
            'InterestRate': self.macro_df['InterestRate'].iloc[-1],
            'GdpGrowth': self.macro_df['GdpGrowth'].iloc[-1],
            'HpiIndex': self.macro_df['HpiIndex'].iloc[-1]
        }
        
        for scenario_name, params in scenarios.items():
            projections = self._project_scenario(last_values.copy(), params['shock'], horizon)
            scenario_projections[scenario_name] = {
                'projections': projections,
                'weight': params['weight'],
                'description': params['description']
            }
            logger.info(f"Scénario {scenario_name} généré - Poids: {params['weight']:.0%}")
        
        return scenario_projections
    
    def _project_scenario(self, current_values: dict, shock: float, horizon: int) -> pd.DataFrame:
        """
        Projeter un scénario avec choc et réversion vers la moyenne
        """
        projections = []
        
        for t in range(1, horizon + 1):
            projected = {}
            
            # Application du choc initial
            if t == 1:
                projected['GdpGrowth'] = current_values['GdpGrowth'] * (1 + shock * 0.15)
                projected['UnemploymentRate'] = current_values['UnemploymentRate'] * (1 - shock * 0.1)
                projected['InterestRate'] = current_values['InterestRate'] * (1 + shock * 0.05)
                projected['HpiIndex'] = current_values['HpiIndex'] * (1 + shock * 0.08)
            else:
                # Réversion progressive vers la moyenne
                reversion_rate = 0.3
                for var in ['UnemploymentRate', 'InterestRate', 'GdpGrowth', 'HpiIndex']:
                    if var in current_values:
                        mean_val = self.macro_df[var].mean()
                        prev_val = projections[t-2][var]
                        projected[var] = prev_val * (1 - reversion_rate) + mean_val * reversion_rate
            
            projected['Year'] = t
            projections.append(projected)
        
        return pd.DataFrame(projections)
    
    def fit_var_model(self, max_lags: int = 4):
        """
        Ajuster un modèle VAR pour les projections
        """
        var_data = self.macro_df[['UnemploymentRate', 'InterestRate', 'GdpGrowth', 'HpiIndex']].copy()
        
        # Différenciation
        for col in ['UnemploymentRate', 'InterestRate', 'GdpGrowth']:
            var_data[f'{col}_Diff'] = var_data[col].diff()
        
        var_data = var_data.dropna()
        
        try:
            model = VAR(var_data[['UnemploymentRate_Diff', 'InterestRate_Diff', 'GdpGrowth_Diff', 'HpiIndex']])
            self.var_model = model.fit(maxlags=max_lags, ic='aic')
            logger.info(f"Modèle VAR ajusté avec {self.var_model.k_ar} lags")
            return self.var_model
        except Exception as e:
            logger.error(f"Erreur VAR: {e}")
            return None