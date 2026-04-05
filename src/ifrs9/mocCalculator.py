import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MOCalculator:
    """
    Calcul des Management Overlays (MO)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_model_uncertainty_overlay(self, base_ecl: float) -> float:
        """
        Calculer un overlay pour l'incertitude du modèle
        """
        # Incertitude basée sur la volatilité des PD
        if 'DefaultFlag' in self.df.columns:
            default_volatility = self.df['DefaultFlag'].std()
            uncertainty_factor = min(0.15, default_volatility * 0.5)
        else:
            uncertainty_factor = 0.05
        
        overlay = base_ecl * uncertainty_factor
        logger.info(f"Overlay incertitude modèle: {overlay:,.0f} ({uncertainty_factor:.1%})")
        
        return overlay
    
    def calculate_scenario_overlay(self, scenario_results: dict) -> float:
        """
        Calculer un overlay basé sur les scénarios
        """
        # Écart entre scénarios
        ecl_values = [v['ecl'] for k, v in scenario_results.items() if k != 'Total']
        
        if len(ecl_values) > 1:
            scenario_spread = np.std(ecl_values)
            scenario_overlay = scenario_spread * 0.2
        else:
            scenario_overlay = 0
        
        logger.info(f"Overlay scénario: {scenario_overlay:,.0f}")
        return scenario_overlay
    
    def calculate_forward_looking_overlay(self, macro_df: pd.DataFrame) -> float:
        """
        Calculer un overlay basé sur les perspectives macro
        """
        if macro_df is not None and len(macro_df) > 0:
            recent_gdp = macro_df['GdpGrowth'].iloc[-3:].mean()
            long_term_gdp = macro_df['GdpGrowth'].mean()
            
            if recent_gdp < long_term_gdp * 0.5:
                overlay_factor = 0.10
            elif recent_gdp < long_term_gdp:
                overlay_factor = 0.05
            else:
                overlay_factor = 0.0
        else:
            overlay_factor = 0.03
        
        logger.info(f"Overlay forward-looking: {overlay_factor:.1%}")
        return overlay_factor
    
    def calculate_total_overlay(self, base_ecl: float, scenario_results: dict, macro_df: pd.DataFrame) -> dict:
        """
        Calculer l'overlay total
        """
        overlays = {
            'model_uncertainty': self.calculate_model_uncertainty_overlay(base_ecl),
            'scenario': self.calculate_scenario_overlay(scenario_results),
            'forward_looking': self.calculate_forward_looking_overlay(macro_df)
        }
        
        total_overlay = sum(overlays.values())
        final_ecl = base_ecl + total_overlay
        
        logger.info(f"Overlay total: {total_overlay:,.0f}")
        logger.info(f"ECL finale avec overlays: {final_ecl:,.0f}")
        
        return {
            'overlays': overlays,
            'total_overlay': total_overlay,
            'final_ecl': final_ecl
        }