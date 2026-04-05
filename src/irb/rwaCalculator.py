import pandas as pd
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

class RWACalculator:
    """
    Calcul des RWA selon formule Bâle IRB
    """
    
    def __init__(self, df: pd.DataFrame, pd_values: pd.Series, lgd_values: pd.Series, ead_values: pd.Series):
        self.df = df
        self.pd_values = pd_values
        self.lgd_values = lgd_values
        self.ead_values = ead_values
        
    def calculate_rwa(self, correlation_rho: float = 0.12, maturity_adjustment: float = 1.0) -> float:
        """
        Calculer les RWA selon formule Bâle IRB
        RWA = 12.5 * EAD * LGD * [Φ( (Φ⁻¹(PD) + √ρ × Φ⁻¹(0.999)) / √(1-ρ) ) - PD]
        """
        rwa_total = 0.0
        
        for idx in self.df.index:
            pd_val = self.pd_values[idx]
            lgd_val = self.lgd_values[idx]
            ead_val = self.ead_values[idx]
            
            if pd_val <= 0 or pd_val >= 1:
                continue
            
            inv_pd = norm.ppf(pd_val)
            inv_percentile = norm.ppf(0.999)
            
            numerator = inv_pd + np.sqrt(correlation_rho) * inv_percentile
            denominator = np.sqrt(1 - correlation_rho)
            
            ul_component = norm.cdf(numerator / denominator)
            
            # Unexpected Loss
            ul = ead_val * lgd_val * (ul_component - pd_val)
            
            # RWA
            rwa = 12.5 * ul * maturity_adjustment
            rwa_total += rwa
        
        logger.info(f"RWA total calculé: {rwa_total:,.0f}")
        return rwa_total
    
    def calculate_el_basel(self) -> float:
        """
        Calculer l'Expected Loss selon Bâle
        """
        el_total = (self.pd_values * self.lgd_values * self.ead_values).sum()
        logger.info(f"EL Bâle: {el_total:,.0f}")
        return el_total
    
    def calculate_capital_requirements(self, rwa: float) -> dict:
        """
        Calculer les exigences de capital
        """
        minimum_ratio = 0.08
        conservation_buffer = 0.025
        
        results = {
            'MinimumCapital': rwa * minimum_ratio,
            'CapitalWithBuffer': rwa * (minimum_ratio + conservation_buffer),
            'Tier1Required': rwa * 0.06,
            'CommonEquityRequired': rwa * 0.045
        }
        
        return results