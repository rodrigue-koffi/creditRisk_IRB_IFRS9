import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Validation complète des modèles de risque de crédit
    """
    
    def __init__(self, y_true: pd.Series, y_pred_proba: pd.Series):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        
    def calculate_all_metrics(self) -> dict:
        """
        Calculer toutes les métriques de validation
        """
        metrics = {
            'AUC': self.calculate_auc(),
            'Gini': self.calculate_gini(),
            'KS': self.calculate_ks(),
            'Accuracy': self.calculate_accuracy(),
            'Precision': self.calculate_precision(),
            'Recall': self.calculate_recall(),
            'Specificity': self.calculate_specificity(),
            'F1_Score': self.calculate_f1_score()
        }
        
        logger.info("Métriques de validation calculées:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def calculate_auc(self) -> float:
        """Calculer l'AUC"""
        return roc_auc_score(self.y_true, self.y_pred_proba)
    
    def calculate_gini(self) -> float:
        """Calculer le Gini = 2*AUC - 1"""
        return 2 * self.calculate_auc() - 1
    
    def calculate_ks(self) -> float:
        """Calculer la statistique KS"""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        ks = max(tpr - fpr)
        return ks
    
    def calculate_accuracy(self, threshold: float = 0.5) -> float:
        """Calculer l'accuracy"""
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        return (self.y_true == y_pred).mean()
    
    def calculate_precision(self, threshold: float = 0.5) -> float:
        """Calculer la précision"""
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        tp = ((self.y_true == 1) & (y_pred == 1)).sum()
        fp = ((self.y_true == 0) & (y_pred == 1)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def calculate_recall(self, threshold: float = 0.5) -> float:
        """Calculer le recall (sensibilité)"""
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        tp = ((self.y_true == 1) & (y_pred == 1)).sum()
        fn = ((self.y_true == 1) & (y_pred == 0)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def calculate_specificity(self, threshold: float = 0.5) -> float:
        """Calculer la spécificité"""
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        tn = ((self.y_true == 0) & (y_pred == 0)).sum()
        fp = ((self.y_true == 0) & (y_pred == 1)).sum()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def calculate_f1_score(self, threshold: float = 0.5) -> float:
        """Calculer le F1-Score"""
        precision = self.calculate_precision(threshold)
        recall = self.calculate_recall(threshold)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    def plot_roc_curve(self, title: str = "Courbe ROC"):
        """
        Tracer la courbe ROC
        """
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        auc = self.calculate_auc()
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
        plt.xlabel('Taux de Faux Positifs (1 - Spécificité)')
        plt.ylabel('Taux de Vrais Positifs (Sensibilité)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=150)
        plt.show()