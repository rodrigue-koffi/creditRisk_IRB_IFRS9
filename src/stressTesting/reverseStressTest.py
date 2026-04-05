import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
import logging

logger = logging.getLogger(__name__)

class ReverseStressTest:
    """
    Reverse Stress Test - Identifier le choc qui provoque un seuil critique
    """
    
    def __init__(self, df: pd.DataFrame, pd_model, lgd_values: pd.Series, ead_values: pd.Series):
        self.df = df
        self.pd_model = pd_model
        self.lgd_values = lgd_values
        self.ead_values = ead_values
        self.stress_engine = None
        
    def find_breakeven_shock(self, target_ecl_multiplier: float = 2.0,
                             capital_ratio_threshold: float = 0.08) -> dict:
        """
        Trouver le chock qui fait atteindre un seuil critique
        """
        from src.stressTesting.stressTestEngine import StressTestEngine
        
        self.stress_engine = StressTestEngine(self.df, self.pd_model, self.lgd_values, self.ead_values)
        
        # Calculer l'ECL de base
        base_ecl = (0.05 * self.lgd_values * self.ead_values).sum()
        
        # Fonction objectif pour trouver le chômage critique
        def ecl_function(unemployment_shock):
            result = self.stress_engine.apply_macro_shock(unemployment_shock=unemployment_shock, gdp_shock=-0.03)
            return result['ecl_stressed'] - base_ecl * target_ecl_multiplier
        
        # Trouver le choc
        try:
            solution = root_scalar(ecl_function, bracket=[0.01, 2.0], method='bisect')
            critical_unemployment = solution.root
        except:
            critical_unemployment = 0.75
        
        logger.info(f"Chômage critique pour multiplier ECL par {target_ecl_multiplier}: {critical_unemployment:.1%}")
        
        # Scénario de reverse stress
        reverse_scenario = {
            'critical_unemployment_shock': critical_unemployment,
            'critical_gdp_shock': -0.05,
            'target_multiplier': target_ecl_multiplier,
            'capital_threshold': capital_ratio_threshold
        }
        
        return reverse_scenario
    
    def identify_weak_segments(self, reverse_scenario: dict) -> pd.DataFrame:
        """
        Identifier les segments les plus vulnérables
        """
        segment_analysis = []
        
        # Analyser par segment d'âge
        age_groups = [(0, 30), (30, 50), (50, 100)]
        
        for age_min, age_max in age_groups:
            segment_mask = (self.df['Age'] >= age_min) & (self.df['Age'] < age_max)
            segment_size = segment_mask.sum()
            
            if segment_size > 0:
                # Stress appliqué
                stress_multiplier = 1 + reverse_scenario['critical_unemployment_shock']
                segment_vulnerability = stress_multiplier * (1 - segment_size / len(self.df))
                
                segment_analysis.append({
                    'Segment': f"{age_min}-{age_max} ans",
                    'Size': segment_size,
                    'Percentage': segment_size / len(self.df),
                    'VulnerabilityScore': segment_vulnerability
                })
        
        # Analyser par type d'emploi
        if 'Job' in self.df.columns:
            for job_type in self.df['Job'].unique():
                segment_mask = self.df['Job'] == job_type
                segment_size = segment_mask.sum()
                
                if segment_size > 0:
                    stress_multiplier = 1 + reverse_scenario['critical_unemployment_shock']
                    segment_vulnerability = stress_multiplier * (1 - segment_size / len(self.df))
                    
                    segment_analysis.append({
                        'Segment': f"Job_{job_type}",
                        'Size': segment_size,
                        'Percentage': segment_size / len(self.df),
                        'VulnerabilityScore': segment_vulnerability
                    })
        
        results_df = pd.DataFrame(segment_analysis).sort_values('VulnerabilityScore', ascending=False)
        logger.info(f"Segments vulnérables identifiés: {len(results_df)} segments")
        
        return results_df
    
    def generate_reverse_stress_report(self) -> dict:
        """
        Générer un rapport complet de reverse stress test
        """
        # Seuils critiques
        breakeven_2x = self.find_breakeven_shock(target_ecl_multiplier=2.0)
        breakeven_3x = self.find_breakeven_shock(target_ecl_multiplier=3.0)
        breakeven_capital = self.find_breakeven_shock(capital_ratio_threshold=0.08)
        
        # Segments vulnérables
        vulnerable_segments = self.identify_weak_segments(breakeven_2x)
        
        report = {
            'breakeven_ecl_2x': breakeven_2x,
            'breakeven_ecl_3x': breakeven_3x,
            'breakeven_capital_threshold': breakeven_capital,
            'vulnerable_segments': vulnerable_segments.to_dict('records'),
            'recommendations': self._generate_recommendations(breakeven_2x, vulnerable_segments)
        }
        
        return report
    
    def _generate_recommendations(self, breakeven: dict, vulnerable_segments: pd.DataFrame) -> list:
        """
        Générer des recommandations basées sur le reverse stress test
        """
        recommendations = []
        
        if breakeven['critical_unemployment_shock'] < 0.30:
            recommendations.append("Renforcer les provisions - vulnérabilité élevée aux chocs modérés")
        
        if len(vulnerable_segments[vulnerable_segments['VulnerabilityScore'] > 1.5]) > 0:
            recommendations.append("Revoir les critères d'octroi pour les segments à haute vulnérabilité")
        
        recommendations.append("Mettre en place un monitoring mensuel des indicateurs macro critiques")
        recommendations.append("Diversifier le portefeuille pour réduire la concentration sectorielle")
        
        return recommendations