import pandas as pd
import numpy as np
from scipy.stats import chi2
import logging

logger = logging.getLogger(__name__)

class CalibrationTest:
    """
    Tests de calibration pour les modèles de PD
    """
    
    def __init__(self, y_true: pd.Series, y_pred_proba: pd.Series, n_bins: int = 10):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.n_bins = n_bins
        
    def hosmer_lemeshow_test(self) -> dict:
        """
        Test de Hosmer-Lemeshow
        """
        # Créer les bins
        self.y_pred_proba = pd.Series(self.y_pred_proba.values, index=self.y_true.index)
        df = pd.DataFrame({'y_true': self.y_true, 'y_pred': self.y_pred_proba})
        df['bin'] = pd.qcut(df['y_pred'], q=self.n_bins, duplicates='drop')
        
        # Calculer les statistiques
        observed = df.groupby('bin')['y_true'].sum()
        expected = df.groupby('bin')['y_pred'].sum()
        n_obs = df.groupby('bin').size()
        
        # Statistique HL
        hl_stat = ((observed - expected) ** 2 / (expected * (1 - expected / n_obs))).sum()
        
        # p-value
        df_hl = self.n_bins - 2
        p_value = 1 - chi2.cdf(hl_stat, df_hl)
        
        result = {
            'HL_Statistic': hl_stat,
            'Degrees_Freedom': df_hl,
            'P_Value': p_value,
            'Well_Calibrated': p_value > 0.05
        }
        
        logger.info(f"Test HL: stat={hl_stat:.4f}, p-value={p_value:.4f}")
        
        return result
    
    def calculate_calibration_curve(self) -> pd.DataFrame:
        """
        Calculer la courbe de calibration
        """
        df = pd.DataFrame({'y_true': self.y_true, 'y_pred': self.y_pred_proba})
        df['bin'] = pd.qcut(df['y_pred'], q=min(self.n_bins, len(df)), duplicates='drop')
        
        calibration_df = df.groupby('bin').agg(
            mean_predicted=('y_pred', 'mean'),
            observed_rate=('y_true', 'mean'),
            count=('y_true', 'count')
        ).reset_index()
        
        calibration_df['bin_center'] = calibration_df['mean_predicted']
        calibration_df['deviation'] = calibration_df['observed_rate'] - calibration_df['mean_predicted']
        calibration_df['abs_deviation'] = np.abs(calibration_df['deviation'])
        
        return calibration_df