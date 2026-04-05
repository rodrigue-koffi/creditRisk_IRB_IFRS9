import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StagingAllocator:
    """
    Allocation Stage 1 / Stage 2 / Stage 3 selon IFRS 9
    """
    
    def __init__(self, df: pd.DataFrame, pd_origination: float, pd_current: float):
        self.df = df
        self.pd_origination = pd_origination
        self.pd_current = pd_current
        
    def allocate_stages(self, relative_threshold: float = 2.0, 
                        absolute_threshold: float = 0.02) -> pd.Series:
        """
        Allouer les stages selon IFRS 9
        """
        # Calcul des indicateurs
        pd_ratio = self.pd_current / max(self.pd_origination, 0.001)
        pd_absolute_diff = self.pd_current - self.pd_origination
        
        # Backstop 30 DPD (simulé)
        np.random.seed(42)
        has_30dpd = np.random.random(len(self.df)) < 0.05
        
        # Détection d'augmentation significative
        significant_increase = (
            (pd_ratio > relative_threshold) | 
            (pd_absolute_diff > absolute_threshold) |
            has_30dpd
        ).astype(int)
        
        # Allocation
        self.df['Stage'] = 1  # Stage 1 par défaut
        self.df.loc[self.df['DefaultFlag'] == 1, 'Stage'] = 3  # Stage 3
        self.df.loc[(self.df['Stage'] == 1) & (significant_increase == 1), 'Stage'] = 2
        
        logger.info("Allocation des stages IFRS 9:")
        for stage, count in self.df['Stage'].value_counts().sort_index().items():
            logger.info(f"  Stage {stage}: {count} prêts ({count/len(self.df):.1%})")
        
        return self.df['Stage']
    
    def get_stage_exposure(self) -> dict:
        """
        Calculer l'exposition par stage
        """
        exposure_by_stage = {}
        for stage in [1, 2, 3]:
            stage_mask = self.df['Stage'] == stage
            exposure_by_stage[stage] = {
                'count': stage_mask.sum(),
                'credit_amount': self.df.loc[stage_mask, 'CreditAmount'].sum(),
                'percentage': stage_mask.mean()
            }
        return exposure_by_stage
    
    def calculate_transition_probabilities(self) -> pd.DataFrame:
        """
        Calculer les probabilités de transition entre stages
        """
        # Simulation de transitions sur 1 an
        np.random.seed(42)
        transition_probs = {
            'FromStage1': {'ToStage1': 0.85, 'ToStage2': 0.12, 'ToStage3': 0.03},
            'FromStage2': {'ToStage1': 0.20, 'ToStage2': 0.60, 'ToStage3': 0.20},
            'FromStage3': {'ToStage1': 0.00, 'ToStage2': 0.00, 'ToStage3': 1.00}
        }
        
        transition_df = pd.DataFrame(transition_probs).T
        logger.info("Matrice de transition entre stages calculée")
        
        return transition_df