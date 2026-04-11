import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import statsmodels.api as sm
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PDOneYearModel:
    def __init__(self, df: pd.DataFrame, target: str = 'DefaultFlag'):
        self.df = df
        self.target = target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.logit_model = None
        self.xgb_model = None
        
        # Stockage des PD
        self.pd_raw = None          # PD brute du modèle (moyenne test)
        self.pd_ttc = None          # PD Through-the-Cycle (Bâle IRB)
        self.pd_pit = None          # PD Point-in-Time (IFRS 9)
        self.pd_pit_weighted = None # PD PIT pondérée IFRS 9
        self.pit_scenarios = None   # Scénarios IFRS 9
        
        # Variables macro
        self.macro_data = None
        self.macro_vars = None
        
    def prepare_data(self, features: list = None):
        if features is None:
            features = ['Age', 'Job', 'Duration', 'CreditAmount', 'SavingAccountsNum', 
                       'CheckingAccountNum', 'HousingNum']
        
        X = self.df[features].copy()
        y = self.df[self.target].copy()
        
        X = X.fillna(X.median())
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # PD brute = taux de défaut moyen sur train
        self.pd_raw = self.y_train.mean()
        
        logger.info(f"Train: {len(self.X_train)} obs, defaults: {self.y_train.sum()}")
        logger.info(f"PD brute (moyenne train): {self.pd_raw:.4f}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def fit_logit(self):
        X_train_const = sm.add_constant(self.X_train)
        self.logit_model = sm.Logit(self.y_train, X_train_const).fit(disp=0)
        logger.info("Logit model fitted")
        return self.logit_model
    
    def fit_xgboost(self, params: dict = None):
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'auc',
                'use_label_encoder': False
            }
        
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight
        
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(self.X_train, self.y_train)
        
        logger.info("XGBoost model fitted")
        return self.xgb_model
    
    def predict_proba(self, model_name: str = 'xgb', dataset: str = 'test') -> np.ndarray:
        if model_name == 'logit' and self.logit_model is not None:
            X = self.X_test if dataset == 'test' else self.X_train
            X_const = sm.add_constant(X)
            return self.logit_model.predict(X_const)
        elif model_name == 'xgb' and self.xgb_model is not None:
            X = self.X_test if dataset == 'test' else self.X_train
            return self.xgb_model.predict_proba(X)[:, 1]
        else:
            raise ValueError(f"Model {model_name} not available")
    
    # PD TTC (Bâle IRB) 
    
    def calculate_pd_ttc(self, method: str = 'simple', year_col: str = 'OriginationYear') -> float:
        """
        Calcule la PD TTC (Through-the-Cycle) pour Bâle IRB
        
        Methods:
        - simple: moyenne historique globale
        - by_year: moyenne des taux annuels (pondération égale)
        - ewm: lissage exponentiel
        - cycle: moyenne sur cycle complet (pic à creux)
        """
        if method == 'simple':
            self.pd_ttc = self.y_train.mean()
            
        elif method == 'by_year' and year_col in self.df.columns:
            annual_rates = self.df.groupby(year_col)[self.target].mean()
            self.pd_ttc = annual_rates.mean()
            logger.info(f"Taux annuels: {annual_rates.to_dict()}")
            
        elif method == 'ewm' and year_col in self.df.columns:
            df_sorted = self.df.sort_values(year_col)
            rolling_defaults = df_sorted[self.target].rolling(window=5, min_periods=1).mean()
            self.pd_ttc = rolling_defaults.iloc[-1]
            
        elif method == 'cycle' and year_col in self.df.columns:
            # Moyenne sur cycle complet (ex: 2008-2013 pour crise financière)
            cycle_years = [2008, 2009, 2010, 2011, 2012, 2013]
            annual_rates = self.df.groupby(year_col)[self.target].mean()
            cycle_rates = annual_rates[annual_rates.index.isin(cycle_years)]
            self.pd_ttc = cycle_rates.mean() if len(cycle_rates) > 0 else self.y_train.mean()
            
        else:
            self.pd_ttc = self.y_train.mean()
        
        self.pd_ttc = min(0.30, max(0.0005, self.pd_ttc))
        
        logger.info(f"PD TTC ({method}): {self.pd_ttc:.4f} ({self.pd_ttc*100:.2f}%)")
        return self.pd_ttc
    
    #  PD PIT (IFRS 9) 
    
    def add_macro_data(self, macro_df: pd.DataFrame, 
                       macro_vars: list = ['UnemploymentRate', 'GdpGrowth']):
        """
        Ajouter des données macroéconomiques pour le calcul de la PD PIT
        """
        self.macro_data = macro_df
        self.macro_vars = macro_vars
        
        # Calcul des valeurs neutres (moyenne cycle)
        self.neutral_values = {}
        for var in macro_vars:
            if var in macro_df.columns:
                self.neutral_values[var] = macro_df[var].mean()
        
        logger.info(f"Données macro ajoutées: {macro_vars}")
        logger.info(f"Valeurs neutres: {self.neutral_values}")
    
    def calculate_pd_pit(self, current_macro_values: Dict[str, float], 
                         macro_sensitivity: float = 0.5) -> float:
        """
        Calcule la PD PIT (Point-in-Time) pour IFRS 9
        
        Formule: PD_PIT = PD_TTC * exp(Σ β_i * (macro_i - macro_neutre_i))
        
        Parameters:
        -----------
        current_macro_values : dict
            Valeurs actuelles des variables macro (ex: {'UnemploymentRate': 8.5})
        macro_sensitivity : float
            Sensibilité (beta) aux chocs macro
        """
        if self.pd_ttc is None:
            self.calculate_pd_ttc()
        
        pit_factor = 1.0
        
        for var, current_val in current_macro_values.items():
            neutral_val = self.neutral_values.get(var, 0)
            deviation = current_val - neutral_val
            
            # Sensibilité différenciée selon la variable
            if var == 'UnemploymentRate':
                beta = macro_sensitivity
            elif var == 'GdpGrowth':
                beta = -macro_sensitivity * 0.8  # Croissance réduit le risque
            else:
                beta = macro_sensitivity * 0.3
            
            pit_factor *= np.exp(beta * deviation)
        
        self.pd_pit = self.pd_ttc * pit_factor
        self.pd_pit = min(0.30, max(0.0005, self.pd_pit))
        
        logger.info(f"PD PIT calculée: {self.pd_pit:.4f} ({self.pd_pit*100:.2f}%)")
        logger.info(f"  PD TTC: {self.pd_ttc:.4f}")
        logger.info(f"  Facteur PIT: {pit_factor:.3f}")
        
        return self.pd_pit
    
    def generate_ifrs9_scenarios(self, horizon: int = 3) -> Dict[str, Dict]:
        """
        Générer les 3 scénarios IFRS 9 (baseline, upside, downside)
        """
        if self.macro_data is None:
            logger.warning("Aucune donnée macro - scénarios basés sur PD TTC")
            self.pit_scenarios = {
                'Baseline': {'pd_pit': self.pd_ttc, 'weight': 0.50},
                'Upside': {'pd_pit': self.pd_ttc * 0.8, 'weight': 0.25},
                'Downside': {'pd_pit': self.pd_ttc * 1.5, 'weight': 0.25}
            }
            return self.pit_scenarios
        
        # Valeurs macro actuelles (dernière année)
        last_macro = {}
        for var in self.macro_vars:
            if var in self.macro_data.columns:
                last_macro[var] = self.macro_data[var].iloc[-1]
        
        # Définition des chocs par scénario
        scenarios_config = {
            'Baseline': {'weight': 0.50, 'shocks': {'UnemploymentRate': 0, 'GdpGrowth': 0}},
            'Upside': {'weight': 0.25, 'shocks': {'UnemploymentRate': -0.5, 'GdpGrowth': 0.5}},
            'Downside': {'weight': 0.25, 'shocks': {'UnemploymentRate': 1.0, 'GdpGrowth': -1.0}}
        }
        
        self.pit_scenarios = {}
        
        for scenario_name, config in scenarios_config.items():
            # Appliquer les chocs
            shocked_macro = {}
            for var in self.macro_vars:
                shock = config['shocks'].get(var, 0)
                shocked_macro[var] = last_macro.get(var, 0) + shock
            
            # Calculer PD PIT pour ce scénario
            pd_pit = self.calculate_pd_pit(shocked_macro, macro_sensitivity=0.5)
            
            self.pit_scenarios[scenario_name] = {
                'pd_pit': pd_pit,
                'weight': config['weight'],
                'description': config['shocks']
            }
        
        # Calculer la PD PIT pondérée
        self.pd_pit_weighted = sum(
            s['pd_pit'] * s['weight'] for s in self.pit_scenarios.values()
        )
        
        logger.info(f"PD PIT pondérée IFRS 9: {self.pd_pit_weighted:.4f}")
        
        return self.pit_scenarios
    
    #  Methode unifié
    
    def calibrate_all_pd(self, macro_df: pd.DataFrame = None, 
                         current_unemployment: float = None) -> Dict[str, Any]:
        """
        Calcule et retourne les 3 types de PD:
        - PD brute (modèle)
        - PD TTC (Bâle IRB)
        - PD PIT (IFRS 9)
        """
        # 1. PD brute
        if self.pd_raw is None:
            self.pd_raw = self.y_train.mean()
        
        # 2. PD TTC
        self.calculate_pd_ttc(method='by_year')
        
        # 3. PD PIT (si macro fournie)
        if macro_df is not None:
            self.add_macro_data(macro_df)
            if current_unemployment is not None:
                self.calculate_pd_pit({'UnemploymentRate': current_unemployment})
            self.generate_ifrs9_scenarios()
        else:
            # Fallback: PD PIT = PD TTC * 1.1
            self.pd_pit = min(0.30, self.pd_ttc * 1.1)
            self.pd_pit_weighted = self.pd_pit
        
        results = {
            'pd_raw': self.pd_raw,
            'pd_ttc': self.pd_ttc,
            'pd_pit': self.pd_pit,
            'pd_pit_weighted': self.pd_pit_weighted if self.pd_pit_weighted else self.pd_pit,
            'ratio_pit_ttc': self.pd_pit / self.pd_ttc if self.pd_ttc > 0 else np.nan
        }
        
        self._print_pd_summary(results)
        
        return results
    
    def _print_pd_summary(self, results: dict):
        """Affiche un résumé des PD calculées"""
        print("\n" + "="*60)
        print("RÉSUMÉ DES PROBABILITÉS DE DÉFAUT (PD 1 an)")
        print("="*60)
        print(f"PD brute du modèle (moyenne train): {results['pd_raw']:.4f} ({results['pd_raw']*100:.2f}%)")
        print(f"PD TTC (Through-the-Cycle) - Bâle IRB: {results['pd_ttc']:.4f} ({results['pd_ttc']*100:.2f}%)")
        print(f"PD PIT (Point-in-Time) - IFRS 9: {results['pd_pit']:.4f} ({results['pd_pit']*100:.2f}%)")
        print(f"PD PIT pondérée IFRS 9: {results['pd_pit_weighted']:.4f} ({results['pd_pit_weighted']*100:.2f}%)")
        print(f"Ratio PIT / TTC: {results['ratio_pit_ttc']:.2f}x")
        print("="*60)
        
        if results['ratio_pit_ttc'] > 1.2:
            print("Contexte dégradé : PD PIT > PD TTC (phase de récession)")
        elif results['ratio_pit_ttc'] < 0.8:
            print("Contexte favorable : PD PIT < PD TTC (phase d'expansion)")
        else:
            print("Contexte neutre : PD PIT proche de PD TTC")
        print("="*60)
    
    #Méthode legacy (conservée pour compatibilité) 
    
    def calibrate_pd_ttc_pit(self) -> tuple:
        """
        Méthode legacy conservée pour compatibilité avec mainOrchestrator
        Retourne (pd_ttc, pd_pit)
        """
        if self.pd_ttc is None:
            self.calculate_pd_ttc('simple')
        if self.pd_pit is None:
            self.pd_pit = min(0.30, max(0.001, self.pd_ttc * 1.1))
        
        logger.info(f"PD TTC (legacy): {self.pd_ttc:.4f}, PD PIT (legacy): {self.pd_pit:.4f}")
        return self.pd_ttc, self.pd_pit