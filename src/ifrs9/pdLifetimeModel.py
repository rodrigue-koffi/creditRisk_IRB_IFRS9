import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
import logging

logger = logging.getLogger(__name__)

class PDLifetimeModel:
    """
    Modélisation de la PD sur la durée de vie (IFRS 9)
    """
    
    def __init__(self, df: pd.DataFrame, pd_1year: float, macro_df: pd.DataFrame):
        self.df = df
        self.pd_1year = pd_1year
        self.macro_df = macro_df
        self.lifetime_df = None
        self.cox_model = None
        
    def create_lifetime_horizon(self, max_years: int = 10) -> pd.DataFrame:
        """
        Créer l'horizon de vie pour chaque prêt
        """
        lifetime_results = []
        
        for idx, row in self.df.iterrows():
            remaining_years = max(1, min(max_years, int(row.get('RemainingTerm', 60) / 12)))
            
            for t in range(1, remaining_years + 1):
                # Projection macro
                base_year = row.get('OriginationYear', 2010)
                future_year = base_year + t
                
                macro_future = self.macro_df[self.macro_df['Year'] == future_year]
                if len(macro_future) > 0:
                    unemployment = macro_future['UnemploymentRate'].iloc[0]
                    gdp = macro_future['GdpGrowth'].iloc[0]
                else:
                    unemployment = self.macro_df['UnemploymentRate'].mean()
                    gdp = self.macro_df['GdpGrowth'].mean()
                
                # PD marginale avec shift macro
                pd_marginal = self.pd_1year * (1 + (unemployment - self.macro_df['UnemploymentRate'].mean()) / 20)
                pd_marginal = np.clip(pd_marginal, 0.001, 0.50)
                
                lifetime_results.append({
                    'Id': idx,
                    'Year': t,
                    'PdMarginal': pd_marginal,
                    'UnemploymentRate': unemployment,
                    'GdpGrowth': gdp
                })
        
        self.lifetime_df = pd.DataFrame(lifetime_results)
        
        # Calculer la survie cumulative
        for idx in self.lifetime_df.index:
            if self.lifetime_df.loc[idx, 'Year'] == 1:
                self.lifetime_df.loc[idx, 'SurvivalProb'] = 1 - self.lifetime_df.loc[idx, 'PdMarginal']
            else:
                prev_survival = self.lifetime_df.loc[idx-1, 'SurvivalProb'] if idx > 0 else 1
                self.lifetime_df.loc[idx, 'SurvivalProb'] = prev_survival * (1 - self.lifetime_df.loc[idx, 'PdMarginal'])
        
        self.lifetime_df['PdCumulative'] = 1 - self.lifetime_df['SurvivalProb']
        
        logger.info(f"Horizon de vie créé: {len(self.lifetime_df)} observations")
        return self.lifetime_df
    
    def fit_cox_model(self):
        """
        Ajuster un modèle de survie Cox PH
        """
        survival_data = self.lifetime_df.copy()
        survival_data['Event'] = 1
        survival_data['Time'] = survival_data['Year']
        
        try:
            self.cox_model = CoxPHFitter()
            self.cox_model.fit(survival_data, duration_col='Time', event_col='Event', 
                               formula='UnemploymentRate + GdpGrowth')
            logger.info("Modèle Cox PH ajusté")
        except Exception as e:
            logger.error(f"Erreur Cox: {e}")
        
        return self.cox_model
    
    def get_marginal_pd_curve(self, loan_id: int) -> pd.DataFrame:
        """
        Obtenir la courbe de PD marginale pour un prêt spécifique
        """
        loan_curve = self.lifetime_df[self.lifetime_df['Id'] == loan_id].copy()
        return loan_curve[['Year', 'PdMarginal', 'PdCumulative', 'SurvivalProb']]