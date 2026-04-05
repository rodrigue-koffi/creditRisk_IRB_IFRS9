import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import statsmodels.api as sm
import logging

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
        
        logger.info(f"Train: {len(self.X_train)} obs, defaults: {self.y_train.sum()}")
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
        
        # Calculer le poids pour les classes déséquilibrées
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight
        
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(self.X_train, self.y_train)
        
        logger.info("XGBoost model fitted")
        return self.xgb_model
    
    def predict_proba(self, model_name: str = 'xgb') -> np.ndarray:
        if model_name == 'logit' and self.logit_model is not None:
            X_test_const = sm.add_constant(self.X_test)
            return self.logit_model.predict(X_test_const)
        elif model_name == 'xgb' and self.xgb_model is not None:
            return self.xgb_model.predict_proba(self.X_test)[:, 1]
        else:
            raise ValueError(f"Model {model_name} not available")
    
    def calibrate_pd_ttc_pit(self) -> tuple:
        pd_ttc = self.y_train.mean()
        pd_pit = min(0.30, max(0.001, pd_ttc * 1.1))
        logger.info(f"PD TTC: {pd_ttc:.4f}, PD PIT: {pd_pit:.4f}")
        return pd_ttc, pd_pit
