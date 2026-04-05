import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ResilienceMetrics:
    """
    Calcul des métriques de résilience du portefeuille
    """
    
    def __init__(self, df: pd.DataFrame, rwa: float, capital: float, ecl: float):
        self.df = df
        self.rwa = rwa
        self.capital = capital
        self.ecl = ecl
        
    def calculate_capital_ratios(self) -> dict:
        """
        Calculer les ratios de capital
        """
        ratios = {
            'CET1_Ratio': self.capital / self.rwa,
            'Tier1_Ratio': self.capital / self.rwa,
            'TotalCapital_Ratio': self.capital / self.rwa,
            'Leverage_Ratio': self.capital / self.df['CreditAmount'].sum()
        }
        
        logger.info(f"CET1 Ratio: {ratios['CET1_Ratio']:.2%}")
        return ratios
    
    def calculate_coverage_ratio(self) -> float:
        """
        Calculer le ratio de couverture (provisions / crédits)
        """
        total_exposure = self.df['CreditAmount'].sum()
        coverage_ratio = self.ecl / total_exposure if total_exposure > 0 else 0
        
        logger.info(f"Coverage Ratio: {coverage_ratio:.2%}")
        return coverage_ratio
    
    def calculate_concentration_metrics(self) -> dict:
        """
        Calculer les métriques de concentration
        """
        # HHI (Herfindahl-Hirschman Index)
        hhi = (self.df.groupby('Purpose')['CreditAmount'].sum() / self.df['CreditAmount'].sum()) ** 2
        hhi_index = hhi.sum()
        
        # Top 10 exposition
        top10_exposure = self.df.nlargest(10, 'CreditAmount')['CreditAmount'].sum()
        top10_ratio = top10_exposure / self.df['CreditAmount'].sum()
        
        metrics = {
            'HHI_Index': hhi_index,
            'Top10_Ratio': top10_ratio,
            'NumberOfPurposeSegments': self.df['Purpose'].nunique()
        }
        
        logger.info(f"HHI Index: {hhi_index:.4f}")
        return metrics
    
    def calculate_stress_buffers(self, stressed_ecl: float) -> dict:
        """
        Calculer les buffers de stress
        """
        buffers = {
            'ECL_Increase_Absolute': stressed_ecl - self.ecl,
            'ECL_Increase_Percentage': (stressed_ecl / self.ecl - 1) if self.ecl > 0 else 0,
            'Capital_Conservation_Buffer': self.capital * 0.025,
            'Stress_Absorption_Capacity': self.capital / stressed_ecl if stressed_ecl > 0 else float('inf')
        }
        
        return buffers