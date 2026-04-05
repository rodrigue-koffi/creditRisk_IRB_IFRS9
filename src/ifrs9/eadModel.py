import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EADModel:
    """
    Modélisation de l'Exposure at Default (EAD)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_ead_committed(self) -> pd.Series:
        """
        Calculer EAD pour produits engagés (prêts)
        """
        # EAD = CreditAmount pour les prêts amortissables
        ead = self.df['CreditAmount'].copy()
        
        logger.info(f"EAD moyenne: {ead.mean():,.0f}")
        return ead
    
    def calculate_ead_uncommitted(self, utilization_rate: float = 0.7) -> pd.Series:
        """
        Calculer EAD pour produits non engagés (découverts)
        """
        # Utilisation simulée
        np.random.seed(42)
        actual_utilization = np.random.uniform(0.3, 1.0, len(self.df))
        
        # EAD = CreditAmount * taux d'utilisation
        ead = self.df['CreditAmount'] * actual_utilization
        
        logger.info(f"EAD non engagé moyenne: {ead.mean():,.0f}")
        return ead
    
    def calculate_ccf(self) -> pd.Series:
        """
        Calculer le Credit Conversion Factor (CCF)
        """
        # Simulation du CCF
        np.random.seed(42)
        ccf = np.random.beta(a=2, b=5, size=len(self.df))
        ccf = ccf.clip(lower=0.1, upper=0.9)
        
        logger.info(f"CCF moyen: {ccf.mean():.2%}")
        return pd.Series(ccf, index=self.df.index)
    
    def calculate_ead_with_ccf(self, ccf: pd.Series) -> pd.Series:
        """
        Calculer EAD avec CCF
        """
        ead = self.df['CreditAmount'] * ccf
        logger.info(f"EAD avec CCF moyenne: {ead.mean():,.0f}")
        return ead