import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import logging

logger = logging.getLogger(__name__)

class DiscriminatoryPower:
    """
    Analyse du pouvoir discriminant des modèles
    """
    
    def __init__(self, models: dict, X_test: pd.DataFrame, y_test: pd.Series):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        
    def compare_all_models(self) -> pd.DataFrame:
        """
        Comparer le pouvoir discriminant de tous les modèles
        """
        results = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, 'predict'):
                y_pred = model.predict(self.X_test)
            else:
                continue
            
            auc = roc_auc_score(self.y_test, y_pred)
            gini = 2 * auc - 1
            
            results.append({
                'Model': model_name,
                'AUC': auc,
                'Gini': gini,
                'Ranking': 'Excellent' if auc > 0.8 else 'Bon' if auc > 0.7 else 'Moyen'
            })
        
        results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
        logger.info("Comparaison des modèles terminée")
        
        return results_df
    
    def calculate_cumulative_accuracy_profile(self, y_pred: pd.Series) -> pd.DataFrame:
        """
        Calculer le Cumulative Accuracy Profile (CAP)
        """
        df = pd.DataFrame({'y_true': self.y_test, 'y_pred': y_pred})
        df = df.sort_values('y_pred', ascending=False)
        df['cumulative_population'] = np.arange(1, len(df) + 1) / len(df)
        df['cumulative_defaults'] = df['y_true'].cumsum() / df['y_true'].sum()
        
        return df[['cumulative_population', 'cumulative_defaults']]