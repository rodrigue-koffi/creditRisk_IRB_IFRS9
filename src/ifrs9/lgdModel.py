import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LGDModel:
    """
    Modélisation de la Loss Given Default (LGD)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_lgd_microstructure(self) -> pd.Series:
        """
        Calculer LGD selon approche micro-structure
        """
        # Facteurs de base
        base_lgd = 0.45
        
        # Collateral
        has_collateral = (self.df.get('HousingNum', 0) == 1).astype(int)
        collateral_reduction = 0.15 * has_collateral
        
        # Impact durée
        duration_impact = 0.005 * (self.df['Duration'] - self.df['Duration'].mean())
        
        # Impact montant
        log_credit = np.log1p(self.df['CreditAmount'])
        amount_impact = 0.02 * (log_credit - log_credit.mean())
        
        # LGD finale
        lgd = base_lgd - collateral_reduction + duration_impact + amount_impact
        lgd = lgd.clip(lower=0.05, upper=0.90)
        
        logger.info(f"LGD moyenne: {lgd.mean():.2%}")
        return lgd
    
    def calculate_lgd_beta_regression(self, features: list = None) -> pd.Series:
        """
        Calculer LGD via régression Beta
        """
        if features is None:
            features = ['Duration', 'CreditAmount', 'Age', 'Job']
        
        # Simulation d'une régression Beta
        beta_score = 0
        for feature in features:
            if feature in self.df.columns:
                normalized = (self.df[feature] - self.df[feature].min()) / (self.df[feature].max() - self.df[feature].min())
                beta_score += normalized
        
        beta_score = beta_score / len(features)
        
        # Transformation en LGD
        lgd = 0.1 + 0.7 * beta_score
        lgd = lgd.clip(lower=0.05, upper=0.90)
        
        logger.info(f"LGD Beta moyenne: {lgd.mean():.2%}")
        return lgd
    
    def calculate_downturn_lgd(self, lgd_normal: pd.Series, downturn_factor: float = 1.2) -> pd.Series:
        """
        Calculer la LGD en période de stress (downturn)
        """
        lgd_downturn = (lgd_normal * downturn_factor).clip(lower=0.05, upper=0.95)
        logger.info(f"LGD Downturn moyenne: {lgd_downturn.mean():.2%}")
        return lgd_downturn