import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataPreparation.dataLoader import DataLoader
from src.dataPreparation.dataCleaner import DataCleaner
from src.dataPreparation.macroDataGenerator import MacroDataGenerator
from src.dataPreparation.featureEngineer import FeatureEngineer
from src.irb.pdOneYearModel import PDOneYearModel
from src.irb.scorecardBuilder import ScorecardBuilder
from src.irb.rwaCalculator import RWACalculator
from src.ifrs9.pdLifetimeModel import PDLifetimeModel
from src.ifrs9.stagingAllocator import StagingAllocator
from src.ifrs9.eclCalculator import ECLCalculator
from src.ifrs9.lgdModel import LGDModel
from src.ifrs9.eadModel import EADModel
from src.ifrs9.mocCalculator import MOCalculator
from src.stressTesting.stressTestEngine import StressTestEngine
from src.stressTesting.reverseStressTest import ReverseStressTest
from src.validation.modelValidator import ModelValidator
from src.utils.logger import setup_logger
from src.utils.helpers import Helpers

logger = setup_logger('MainOrchestrator', log_file='logs/project.log')

class MainOrchestrator:
    """
    Orchestrateur principal du projet de modélisation du risque de crédit
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.results = {}
        
    def run_full_pipeline(self) -> dict:
        """
        Exécuter l'intégralité du pipeline
        """
        logger.info("=" * 80)
        logger.info("DÉMARRAGE DU PIPELINE DE MODÉLISATION DU RISQUE DE CRÉDIT")
        logger.info("=" * 80)
        
        # Phase 1: Chargement et préparation des données
        self._phase1_data_preparation()
        
        # Phase 2: PD 1 an (Bâle IRB)
        self._phase2_pd_1year()
        
        # Phase 3: PD Lifetime (IFRS 9)
        self._phase3_pd_lifetime()
        
        # Phase 4: LGD et EAD
        self._phase4_lgd_ead()
        
        # Phase 5: Staging IFRS 9
        self._phase5_staging()
        
        # Phase 6: ECL et scénarios
        self._phase6_ecl_scenarios()
        
        # Phase 7: Stress test et reverse stress test
        self._phase7_stress_testing()
        
        # Phase 8: Validation
        self._phase8_validation()
        
        # Phase 9: Résultats finaux
        self._phase9_final_results()
        
        logger.info("=" * 80)
        logger.info("PIPELINE TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 80)
        
        return self.results
    
    def _phase1_data_preparation(self):
        """Phase 1: Chargement et préparation des données"""
        logger.info("\n[PHASE 1] Préparation des données")
        
        # Chargement
        loader = DataLoader(self.data_path)
        raw_data = loader.load_german_credit_data()
        macro_data = loader.load_macro_data()
        
        # Nettoyage
        cleaner = DataCleaner(raw_data)
        cleaned_data = cleaner.clean()
        
        # Fusion macro
        np.random.seed(42)
        cleaned_data['OriginationYear'] = np.random.choice(macro_data['Year'].values, len(cleaned_data))
        cleaned_data = cleaned_data.merge(macro_data, left_on='OriginationYear', right_on='Year', how='left')
        
        # Feature engineering
        engineer = FeatureEngineer(cleaned_data)
        self.df = engineer.create_all_features()
        
        self.macro_df = macro_data
        self.results['phase1'] = {
            'data_shape': self.df.shape,
            'default_rate': self.df['DefaultFlag'].mean()
        }
        
        logger.info(f"  Données préparées: {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
    
    def _phase2_pd_1year(self):
        """Phase 2: PD 1 an Bâle IRB"""
        logger.info("\n[PHASE 2] PD 1 an (Bâle IRB)")
        
        # Modèle PD
        self.pd_model = PDOneYearModel(self.df)
        self.pd_model.prepare_data()
        self.pd_model.fit_logit()
        self.pd_model.fit_xgboost()
        
        # Scorecard
        scorecard = ScorecardBuilder(self.pd_model.X_train, self.pd_model.y_train)
        features = self.pd_model.X_train.columns.tolist()
        iv_df = scorecard.compute_all_iv(features)
        selected_features = scorecard.select_features_by_iv()
        
        # Calibration PD
        self.pd_ttc, self.pd_pit = self.pd_model.calibrate_pd_ttc_pit()
        
        # Prédictions
        self.y_pred_logit = self.pd_model.predict_proba('logit')
        self.y_pred_xgb = self.pd_model.predict_proba('xgb')
        
        self.results['phase2'] = {
            'pd_ttc': self.pd_ttc,
            'pd_pit': self.pd_pit,
            'selected_features': selected_features,
            'iv_summary': iv_df.to_dict('records')
        }
        
        logger.info(f"  PD TTC: {self.pd_ttc:.4f}, PD PIT: {self.pd_pit:.4f}")
        logger.info(f"  Features sélectionnées: {len(selected_features)}")
    
    def _phase3_pd_lifetime(self):
        """Phase 3: PD Lifetime IFRS 9"""
        logger.info("\n[PHASE 3] PD Lifetime (IFRS 9)")
        
        self.lifetime_model = PDLifetimeModel(self.df, self.pd_pit, self.macro_df)
        self.lifetime_df = self.lifetime_model.create_lifetime_horizon()
        self.lifetime_model.fit_cox_model()
        
        self.results['phase3'] = {
            'lifetime_observations': len(self.lifetime_df),
            'avg_pd_year1': self.lifetime_df[self.lifetime_df['Year'] == 1]['PdMarginal'].mean(),
            'avg_pd_year5': self.lifetime_df[self.lifetime_df['Year'] == 5]['PdMarginal'].mean() if len(self.lifetime_df[self.lifetime_df['Year'] == 5]) > 0 else None
        }
    
    def _phase4_lgd_ead(self):
        """Phase 4: LGD et EAD"""
        logger.info("\n[PHASE 4] LGD et EAD")
        
        lgd_model = LGDModel(self.df)
        self.lgd_values = lgd_model.calculate_lgd_microstructure()
        
        ead_model = EADModel(self.df)
        self.ead_values = ead_model.calculate_ead_committed()
        
        self.results['phase4'] = {
            'lgd_mean': self.lgd_values.mean(),
            'lgd_std': self.lgd_values.std(),
            'ead_mean': self.ead_values.mean(),
            'ead_total': self.ead_values.sum()
        }
        
        logger.info(f"  LGD moyenne: {self.lgd_values.mean():.2%}")
        logger.info(f"  EAD totale: {self.ead_values.sum():,.0f}")
    
    def _phase5_staging(self):
        """Phase 5: Staging IFRS 9"""
        logger.info("\n[PHASE 5] Staging IFRS 9")
        
        self.staging = StagingAllocator(self.df, self.pd_ttc, self.pd_pit)
        self.df['Stage'] = self.staging.allocate_stages()
        self.stage_exposure = self.staging.get_stage_exposure()
        
        self.results['phase5'] = {
            'stage_allocation': {k: v['count'] for k, v in self.stage_exposure.items()},
            'stage_exposure': {k: v['credit_amount'] for k, v in self.stage_exposure.items()}
        }
    
    def _phase6_ecl_scenarios(self):
        """Phase 6: ECL et scénarios IFRS 9"""
        logger.info("\n[PHASE 6] ECL et scénarios")
        
        # Génération des scénarios
        scenario_gen = MacroDataGenerator(self.macro_df)
        self.scenarios = scenario_gen.generate_scenarios()
        
        # Calcul ECL
        self.pd_1year_series = pd.Series([self.pd_pit] * len(self.df), index=self.df.index)
        
        ecl_calc = ECLCalculator(self.df, self.pd_1year_series, self.lgd_values, self.ead_values, self.lifetime_df)
        
        # ECL par stage
        self.ecl_by_stage, self.total_ecl = ecl_calc.calculate_ecl_by_stage()
        
        # ECL par scénario
        self.ecl_by_scenario = ecl_calc.calculate_ecl_by_scenario(self.scenarios)
        
        # Management Overlays
        moc_calc = MOCalculator(self.df)
        self.moc_results = moc_calc.calculate_total_overlay(self.total_ecl, self.ecl_by_scenario, self.macro_df)
        
        self.results['phase6'] = {
            'ecl_by_stage': self.ecl_by_stage,
            'total_ecl': self.total_ecl,
            'final_ecl_with_moc': self.moc_results['final_ecl'],
            'ecl_by_scenario': {k: v['ecl'] for k, v in self.ecl_by_scenario.items() if k != 'Total'}
        }
        
        logger.info(f"  ECL totale: {self.total_ecl:,.0f}")
        logger.info(f"  ECL finale avec MOC: {self.moc_results['final_ecl']:,.0f}")
    
    def _phase7_stress_testing(self):
        """Phase 7: Stress test et reverse stress test"""
        logger.info("\n[PHASE 7] Stress Testing")
        
        # Stress test
        stress_engine = StressTestEngine(self.df, self.pd_model, self.lgd_values, self.ead_values)
        self.stress_results = stress_engine.run_severity_scenarios()
        
        # Reverse stress test
        reverse_engine = ReverseStressTest(self.df, self.pd_model, self.lgd_values, self.ead_values)
        self.reverse_results = reverse_engine.generate_reverse_stress_report()
        
        self.results['phase7'] = {
            'stress_scenarios': self.stress_results,
            'reverse_stress': {
                'critical_unemployment': self.reverse_results['breakeven_ecl_2x']['critical_unemployment_shock'],
                'vulnerable_segments': len(self.reverse_results['vulnerable_segments']),
                'recommendations': self.reverse_results['recommendations']
            }
        }
        
        logger.info(f"  Chômage critique (ECL x2): {self.reverse_results['breakeven_ecl_2x']['critical_unemployment_shock']:.1%}")
    
    def _phase8_validation(self):
        """Phase 8: Validation des modèles"""
        logger.info("\n[PHASE 8] Validation des modèles")
        
        # Validation du modèle XGBoost
        validator_xgb = ModelValidator(self.pd_model.y_test, self.y_pred_xgb)
        self.xgb_metrics = validator_xgb.calculate_all_metrics()
        validator_xgb.plot_roc_curve("ROC Curve - XGBoost Model")
        
        # Validation du modèle Logit
        validator_logit = ModelValidator(self.pd_model.y_test, self.y_pred_logit)
        self.logit_metrics = validator_logit.calculate_all_metrics()
        
        self.results['phase8'] = {
            'xgb_model': self.xgb_metrics,
            'logit_model': self.logit_metrics,
            'best_model': 'XGBoost' if self.xgb_metrics['AUC'] > self.logit_metrics['AUC'] else 'Logit'
        }
        
        logger.info(f"  XGBoost - AUC: {self.xgb_metrics['AUC']:.4f}, Gini: {self.xgb_metrics['Gini']:.4f}")
        logger.info(f"  Logit - AUC: {self.logit_metrics['AUC']:.4f}, Gini: {self.logit_metrics['Gini']:.4f}")
    
    def _phase9_final_results(self):
        """Phase 9: Résultats finaux et sauvegarde"""
        logger.info("\n[PHASE 9] Résultats finaux")
        
        # Calcul RWA et capital
        rwa_calc = RWACalculator(self.df, self.pd_1year_series, self.lgd_values, self.ead_values)
        self.rwa = rwa_calc.calculate_rwa()
        self.el_basel = rwa_calc.calculate_el_basel()
        
        # Shortfall
        shortfall = max(0, self.el_basel - self.total_ecl)
        
        self.results['phase9'] = {
            'rwa_basel': self.rwa,
            'el_basel': self.el_basel,
            'shortfall': shortfall,
            'capital_impact': shortfall / 1000000 if shortfall > 0 else 0
        }
        
        # Sauvegarde des résultats
        output_dir = Helpers.create_output_directory()
        Helpers.save_results(self.results, output_dir / 'final_results.json')
        
        logger.info(f"  RWA: {self.rwa:,.0f}")
        logger.info(f"  Shortfall: {shortfall:,.0f}")
        logger.info(f"  Résultats sauvegardés dans: {output_dir}")

# Point d'entrée principal
if __name__ == "__main__":
    orchestrator = MainOrchestrator(data_path='data')
    final_results = orchestrator.run_full_pipeline()
    
    print("\n" + "=" * 80)
    print("PROJET TERMINÉ - RÉSULTATS DISPONIBLES")
    print("=" * 80)
    print(f"  PD TTC: {orchestrator.pd_ttc:.4f}")
    print(f"  PD PIT: {orchestrator.pd_pit:.4f}")
    print(f"  ECL totale: {orchestrator.total_ecl:,.0f}")
    print(f"  RWA: {orchestrator.rwa:,.0f}")
    print(f"  Shortfall: {max(0, orchestrator.el_basel - orchestrator.total_ecl):,.0f}")