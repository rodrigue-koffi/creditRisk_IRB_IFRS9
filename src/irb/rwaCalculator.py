import pandas as pd
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

class RWACalculator:
    """
    Calcul des RWA selon formule Bâle IRB
    Utilise PD TTC (cycle long) pour capital réglementaire
    """
    
    def __init__(self, df: pd.DataFrame, pd_values: pd.Series, lgd_values: pd.Series, ead_values: pd.Series):
        self.df = df
        self.pd_values = pd_values
        self.lgd_values = lgd_values
        self.ead_values = ead_values
        
    def calculate_rwa(self, correlation_rho: float = 0.12, maturity_adjustment: float = 1.0) -> float:
        """
        Calculer les RWA selon formule Bâle IRB
        Utilise PD TTC (cycle long) pour capital réglementaire
        
        Formule Bâle IRB:
        RWA = 12.5 * EAD * LGD * [Φ( (Φ⁻¹(PD) + √ρ × Φ⁻¹(0.999)) / √(1-ρ) ) - PD]
        """
        rwa_total = 0.0
        
        for idx in self.df.index:
            pd_val = self.pd_values[idx]
            
            # Bornes réglementaires PD (Bâle IRB)
            pd_val = min(0.30, max(0.0003, pd_val))
            
            lgd_val = self.lgd_values[idx]
            ead_val = self.ead_values[idx]
            
            if pd_val <= 0 or pd_val >= 1:
                continue
            
            # Formule Bâle
            inv_pd = norm.ppf(pd_val)
            inv_percentile = norm.ppf(0.999)  # 99.9% confiance
            
            numerator = inv_pd + np.sqrt(correlation_rho) * inv_percentile
            denominator = np.sqrt(1 - correlation_rho)
            
            ul_component = norm.cdf(numerator / denominator)
            
            # Unexpected Loss (UL)
            ul = ead_val * lgd_val * (ul_component - pd_val)
            
            # RWA
            rwa = 12.5 * ul * maturity_adjustment
            rwa_total += rwa
        
        logger.info(f"RWA total calculé (avec PD TTC): {rwa_total:,.0f}")
        return rwa_total
    
    def calculate_el_basel(self) -> float:
        """
        Calculer l'Expected Loss selon Bâle
        EL = PD * LGD * EAD
        """
        el_total = (self.pd_values * self.lgd_values * self.ead_values).sum()
        logger.info(f"EL Bâle: {el_total:,.0f}")
        return el_total
    
    def calculate_capital_requirements(self, rwa: float) -> dict:
        """
        Calculer les exigences de capital selon Bâle III
        
        Parameters:
        -----------
        rwa : Risk-Weighted Assets (RWA)
        """
        minimum_ratio = 0.08          # 8% minimum
        conservation_buffer = 0.025   # 2.5% buffer de conservation
        countercyclical_buffer = 0.025  # 2.5% buffer contra-cyclique (optionnel)
        
        results = {
            'MinimumCapital': rwa * minimum_ratio,
            'CapitalWithConservationBuffer': rwa * (minimum_ratio + conservation_buffer),
            'TotalCapitalRequirement': rwa * (minimum_ratio + conservation_buffer + countercyclical_buffer),
            'Tier1Required': rwa * 0.06,      # 6% Tier 1
            'CommonEquityRequired': rwa * 0.045  # 4.5% Common Equity Tier 1
        }
        
        logger.info(f"Capital minimum requis: {results['MinimumCapital']:,.0f}")
        logger.info(f"Capital total avec buffers: {results['TotalCapitalRequirement']:,.0f}")
        
        return results
    
    def calculate_rwa_by_segment(self, segment_col: str = 'Purpose') -> pd.DataFrame:
        """
        Calculer les RWA par segment (pour analyse de concentration)
        """
        results = []
        
        for segment, group in self.df.groupby(segment_col):
            # Sous-ensemble des valeurs pour ce segment
            pd_segment = self.pd_values[group.index]
            lgd_segment = self.lgd_values[group.index]
            ead_segment = self.ead_values[group.index]
            
            # Calcul RWA pour ce segment
            rwa_segment = 0
            for idx in group.index:
                pd_val = min(0.30, max(0.0003, self.pd_values[idx]))
                lgd_val = self.lgd_values[idx]
                ead_val = self.ead_values[idx]
                
                if pd_val > 0 and pd_val < 1:
                    inv_pd = norm.ppf(pd_val)
                    inv_percentile = norm.ppf(0.999)
                    numerator = inv_pd + np.sqrt(0.12) * inv_percentile
                    denominator = np.sqrt(1 - 0.12)
                    ul_component = norm.cdf(numerator / denominator)
                    ul = ead_val * lgd_val * (ul_component - pd_val)
                    rwa_segment += 12.5 * ul
            
            results.append({
                'Segment': segment,
                'Count': len(group),
                'Exposure': ead_segment.sum(),
                'RWA': rwa_segment,
                'RWA_Density': rwa_segment / ead_segment.sum() if ead_segment.sum() > 0 else 0
            })
        
        rwa_by_segment_df = pd.DataFrame(results).sort_values('RWA', ascending=False)
        
        logger.info("RWA par segment calculé")
        return rwa_by_segment_df