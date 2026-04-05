import pandas as pd
import numpy as np
from scipy.stats import beta
import logging

logger = logging.getLogger(__name__)

class CCFModel:
    """
    Modélisation du Credit Conversion Factor (CCF)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def fit_beta_distribution(self) -> dict:
        """
        Ajuster une distribution Beta au CCF
        """
        # Simulation de CCF historiques
        np.random.seed(42)
        ccf_historical = np.random.beta(a=2, b=5, size=1000)
        
        # Ajustement des paramètres Beta
        alpha, beta_param, loc, scale = beta.fit(ccf_historical)
        
        logger.info(f"Paramètres Beta: alpha={alpha:.4f}, beta={beta_param:.4f}")
        
        return {
            'alpha': alpha,
            'beta': beta_param,
            'loc': loc,
            'scale': scale
        }
    
    def calculate_expected_ccf(self, params: dict = None) -> pd.Series:
        """
        Calculer le CCF attendu
        """
        if params is None:
            params = {'alpha': 2.0, 'beta': 5.0}
        
        # CCF attendu = alpha / (alpha + beta)
        expected_ccf = params['alpha'] / (params['alpha'] + params['beta'])
        
        # Variation par segment
        segment_variation = 1.0
        if 'Job' in self.df.columns:
            segment_variation = 0.8 + 0.1 * (self.df['Job'] / self.df['Job'].max())
        
        ccf = expected_ccf * segment_variation
        ccf = ccf.clip(lower=0.1, upper=0.9)
        
        return ccf