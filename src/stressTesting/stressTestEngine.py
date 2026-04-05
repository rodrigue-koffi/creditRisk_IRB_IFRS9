import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StressTestEngine:
    """
    Moteur de stress test pour le risque de crédit
    """
    
    def __init__(self, df: pd.DataFrame, pd_model, lgd_values: pd.Series, ead_values: pd.Series):
        self.df = df
        self.pd_model = pd_model
        self.lgd_values = lgd_values
        self.ead_values = ead_values
        
    def apply_macro_shock(self, unemployment_shock: float = 0.30,
                          gdp_shock: float = -0.05,
                          interest_shock: float = 0.02) -> pd.DataFrame:
        """
        Appliquer un choc macroéconomique
        """
        stressed_df = self.df.copy()
        
        # Appliquer les chocs
        if 'UnemploymentRate' in stressed_df.columns:
            stressed_df['UnemploymentRate_Stressed'] = stressed_df['UnemploymentRate'] * (1 + unemployment_shock)
        
        if 'GdpGrowth' in stressed_df.columns:
            stressed_df['GdpGrowth_Stressed'] = stressed_df['GdpGrowth'] + gdp_shock
        
        # Calculer les PD stressées
        pd_stressed = self._calculate_stressed_pd(stressed_df, unemployment_shock, gdp_shock)
        
        # LGD stressée
        lgd_stressed = self.lgd_values * (1 + unemployment_shock * 0.3)
        lgd_stressed = lgd_stressed.clip(lower=0.05, upper=0.95)
        
        # Calculer ECL stressée
        ecl_stressed = (pd_stressed * lgd_stressed * self.ead_values).sum()
        
        logger.info(f"Stress test - ECL stressée: {ecl_stressed:,.0f}")
        logger.info(f"  Chômage: +{unemployment_shock:.0%}, PIB: {gdp_shock:.1%}, Taux: +{interest_shock:.0%}")
        
        return {
            'pd_stressed': pd_stressed,
            'lgd_stressed': lgd_stressed,
            'ecl_stressed': ecl_stressed,
            'stressed_df': stressed_df
        }
    
    def _calculate_stressed_pd(self, df: pd.DataFrame, unemployment_shock: float, gdp_shock: float) -> pd.Series:
        """
        Calculer les PD sous stress
        """
        # Facteur de stress basé sur les chocs
        stress_factor = 1 + unemployment_shock * 2 + max(0, -gdp_shock * 3)
        stress_factor = min(5.0, stress_factor)
        
        # Appliquer le stress à la PD de base
        if self.pd_model is not None and hasattr(self.pd_model, 'pd_base'):
            pd_stressed = self.pd_model.pd_base * stress_factor
        else:
            pd_stressed = pd.Series([0.05] * len(df), index=df.index) * stress_factor
        
        pd_stressed = pd_stressed.clip(lower=0.001, upper=0.60)
        
        return pd_stressed
    
    def run_severity_scenarios(self) -> dict:
        """
        Exécuter plusieurs scénarios de sévérité
        """
        scenarios = {
            'Light': {'unemployment': 0.10, 'gdp': -0.02, 'interest': 0.01},
            'Moderate': {'unemployment': 0.25, 'gdp': -0.05, 'interest': 0.02},
            'Severe': {'unemployment': 0.50, 'gdp': -0.10, 'interest': 0.03},
            'Extreme': {'unemployment': 1.00, 'gdp': -0.15, 'interest': 0.05}
        }
        
        results = {}
        
        for scenario_name, shocks in scenarios.items():
            result = self.apply_macro_shock(
                unemployment_shock=shocks['unemployment'],
                gdp_shock=shocks['gdp'],
                interest_shock=shocks['interest']
            )
            results[scenario_name] = {
                'shocks': shocks,
                'ecl_stressed': result['ecl_stressed'],
                'pd_multiplier': result['pd_stressed'].mean() / 0.05 if hasattr(self, 'pd_model') else 1
            }
        
        return results