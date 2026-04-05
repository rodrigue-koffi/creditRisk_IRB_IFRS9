import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)

class PDOneYearModel:
    """
    Modélisation de la PD à 1 an (Bâle IRB)
    """
    
    def __init__(self, df: pd.DataFrame, target: str = 'DefaultFlag'):
        self.df = df
        self.target = target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.logit_model = None
        self.xgb_model = None
        
    def prepare_data(self, features: list = None):
        """
        Préparer les données pour la modélisation
        """
        if features is None:
            features = ['Age', 'Job', 'HousingNum', 'SavingAccountsNum', 'CheckingAccountNum',
                       'LogCreditAmount', 'Duration', 'TimeOnBook', 'RemainingTerm']
        
        X = self.df[features].copy()
        y = self.df[self.target].copy()
        
        # Gérer les valeurs manquantes
        X = X.fillna(X.median())
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        logger.info(f"Train: {len(self.X_train)} obs, {self.y_train.sum()} défauts")
        logger.info(f"Test: {len(self.X_test)} obs, {self.y_test.sum()} défauts")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def fit_logit(self):
        """
        Ajuster le modèle Logit
        """
        X_train_const = sm.add_constant(self.X_train)
        self.logit_model = sm.Logit(self.y_train, X_train_const).fit(disp=0)
        logger.info("Modèle Logit ajusté")
        return self.logit_model
    
    def fit_xgboost(self, params: dict = None):
        """
        Ajuster le modèle XGBoost
        """
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': (self.y_train == 0).sum() / (self.y_train == 1).sum(),
                'random_state': 42
            }
        
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(self.X_train, self.y_train, 
                           eval_set=[(self.X_test, self.y_test)],
                           early_stopping_rounds=20,
                           verbose=False)
        
        logger.info("Modèle XGBoost ajusté")
        return self.xgb_model
    
    def predict_proba(self, model_name: str = 'xgb') -> np.ndarray:
        """
        Prédire les probabilités
        """
        if model_name == 'logit' and self.logit_model is not None:
            X_test_const = sm.add_constant(self.X_test)
            return self.logit_model.predict(X_test_const)
        elif model_name == 'xgb' and self.xgb_model is not None:
            return self.xgb_model.predict_proba(self.X_test)[:, 1]
        else:
            raise ValueError(f"Modèle {model_name} non disponible")
    
    def calibrate_pd_ttc_pit(self) -> tuple:
        """
        Calibrer PD TTC (Bâle) et PIT (IFRS 9)
        """
        # PD TTC : moyenne historique
        pd_ttc = self.y_train.mean()
        
        # PD PIT : ajustement macro
        if 'UnemploymentRate' in self.df.columns:
            latest_unemployment = self.df['UnemploymentRate'].iloc[-1]
            mean_unemployment = self.df['UnemploymentRate'].mean()
            pit_adjustment = 1 + (latest_unemployment - mean_unemployment) / 20
        else:
            pit_adjustment = 1.0
        
        pd_pit = min(0.30, max(0.001, pd_ttc * pit_adjustment))
        
        logger.info(f"PD TTC: {pd_ttc:.4f}, PD PIT: {pd_pit:.4f}")
        return pd_ttc, pd_pit