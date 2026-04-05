import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ScorecardBuilder:
    """
    Construction du scorecard (WOE, IV, binning)
    """
    
    def __init__(self, df: pd.DataFrame, target: str = 'DefaultFlag'):
        self.df = df
        self.target = target
        self.woe_mappings = {}
        self.iv_values = {}
        
    def calculate_woe_iv(self, feature: str, n_bins: int = 10) -> tuple:
        """
        Calculer WOE et IV pour une variable
        """
        df_temp = pd.DataFrame({feature: self.df[feature], 'target': self.df[self.target]})
        
        try:
            df_temp['bin'] = pd.qcut(df_temp[feature], q=n_bins, duplicates='drop')
        except:
            df_temp['bin'] = pd.cut(df_temp[feature], bins=n_bins)
        
        grouped = df_temp.groupby('bin').agg(
            total=('target', 'count'),
            bad=('target', 'sum'),
            good=('target', lambda x: (x == 0).sum())
        ).reset_index()
        
        grouped['bad'] = grouped['bad'].clip(lower=0.5)
        grouped['good'] = grouped['good'].clip(lower=0.5)
        
        grouped['dist_good'] = grouped['good'] / grouped['good'].sum()
        grouped['dist_bad'] = grouped['bad'] / grouped['bad'].sum()
        grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
        grouped['iv_contrib'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']
        
        iv = grouped['iv_contrib'].sum()
        
        return grouped[['bin', 'woe']], iv
    
    def compute_all_iv(self, features: list) -> pd.DataFrame:
        """
        Calculer l'IV pour toutes les variables
        """
        results = []
        for feature in features:
            try:
                _, iv = self.calculate_woe_iv(feature)
                self.iv_values[feature] = iv
                results.append({'Feature': feature, 'IV': iv})
            except Exception as e:
                logger.warning(f"Erreur pour {feature}: {e}")
                results.append({'Feature': feature, 'IV': 0})
        
        iv_df = pd.DataFrame(results).sort_values('IV', ascending=False)
        
        # Classification IV
        iv_df['PredictivePower'] = iv_df['IV'].apply(
            lambda x: 'Strong' if x > 0.3 else 'Medium' if x > 0.1 else 'Weak'
        )
        
        logger.info("IV calculés pour toutes les variables")
        return iv_df
    
    def select_features_by_iv(self, iv_threshold: float = 0.1) -> list:
        """
        Sélectionner les features avec IV > seuil
        """
        selected = [f for f, iv in self.iv_values.items() if iv > iv_threshold]
        logger.info(f"Features sélectionnées: {len(selected)} / {len(self.iv_values)}")
        return selected
    
    def compute_score(self, coefficients: dict, base_score: float = 600, pdo: int = 20) -> pd.Series:
        """
        Calculer le score à partir des coefficients
        """
        score = base_score + pdo * np.log(1/0.5) * coefficients.get('const', 0)
        for feature, coeff in coefficients.items():
            if feature != 'const' and feature in self.df.columns:
                score += coeff * self.df[feature]
        
        return score