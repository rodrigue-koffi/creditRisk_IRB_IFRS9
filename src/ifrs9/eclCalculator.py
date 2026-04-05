import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ECLCalculator:
    """
    Calcul des Expected Credit Losses (IFRS 9)
    """
    
    def __init__(self, df: pd.DataFrame, pd_1year: pd.Series, 
                 lgd_values: pd.Series, ead_values: pd.Series,
                 lifetime_df: pd.DataFrame):
        self.df = df
        self.pd_1year = pd_1year
        self.lgd_values = lgd_values
        self.ead_values = ead_values
        self.lifetime_df = lifetime_df
        
    def calculate_ecl_by_stage(self, discount_rate: float = 0.03) -> dict:
        """
        Calculer l'ECL par stage
        """
        ecl_by_stage = {}
        total_ecl = 0
        
        for stage in [1, 2, 3]:
            stage_mask = self.df['Stage'] == stage
            stage_ecl = 0
            
            for idx in self.df[stage_mask].index:
                if stage == 1:
                    # Stage 1: ECL sur 1 an
                    loan_ecl = self.pd_1year[idx] * self.lgd_values[idx] * self.ead_values[idx]
                else:
                    # Stage 2/3: ECL lifetime
                    loan_data = self.lifetime_df[self.lifetime_df['Id'] == idx]
                    loan_ecl = 0
                    for _, row in loan_data.iterrows():
                        discount_factor = 1 / (1 + discount_rate) ** row['Year']
                        loan_ecl += row['PdMarginal'] * self.lgd_values[idx] * self.ead_values[idx] * discount_factor
                
                stage_ecl += loan_ecl
            
            ecl_by_stage[stage] = stage_ecl
            total_ecl += stage_ecl
            
            logger.info(f"Stage {stage} ECL: {stage_ecl:,.0f}")
        
        logger.info(f"ECL totale: {total_ecl:,.0f}")
        return ecl_by_stage, total_ecl
    
    def calculate_ecl_by_scenario(self, scenario_projections: dict, discount_rate: float = 0.03) -> dict:
        """
        Calculer l'ECL par scénario IFRS 9
        """
        results = {}
        
        for scenario_name, scenario_data in scenario_projections.items():
            projections = scenario_data['projections']
            weight = scenario_data['weight']
            scenario_ecl = 0
            
            for idx in self.df.index:
                loan_ecl = 0
                if self.df.loc[idx, 'Stage'] == 1:
                    loan_ecl = self.pd_1year[idx] * self.lgd_values[idx] * self.ead_values[idx]
                else:
                    for t, row in projections.iterrows():
                        discount_factor = 1 / (1 + discount_rate) ** row['Year']
                        pd_adjusted = self.pd_1year[idx] * (1 + (row['UnemploymentRate'] - 5) / 20)
                        pd_adjusted = np.clip(pd_adjusted, 0.001, 0.50)
                        loan_ecl += pd_adjusted * self.lgd_values[idx] * self.ead_values[idx] * discount_factor
                
                scenario_ecl += loan_ecl
            
            results[scenario_name] = {
                'ecl': scenario_ecl,
                'weight': weight,
                'weighted_ecl': scenario_ecl * weight
            }
            
            logger.info(f"Scénario {scenario_name}: ECL={scenario_ecl:,.0f}, Poids={weight:.0%}")
        
        total_ecl = sum(r['weighted_ecl'] for r in results.values())
        results['Total'] = {'ecl': total_ecl, 'weight': 1.0, 'weighted_ecl': total_ecl}
        
        return results