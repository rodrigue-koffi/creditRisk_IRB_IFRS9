import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Nettoyage et transformation des données
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def clean(self) -> pd.DataFrame:
        """
        Nettoyage principal des données
        """
        self._rename_columns()
        self._encode_target()
        self._encode_categorical()
        self._create_ratios()
        self._handle_missing_values()
        return self.df
    
    def _rename_columns(self):
        """Renommer les colonnes en CamelCase"""
        self.df.columns = ['Age', 'Sex', 'Job', 'Housing', 'SavingAccounts', 
                           'CheckingAccount', 'CreditAmount', 'Duration', 'Purpose', 'Risk']
        logger.info("Colonnes renommées")
    
    def _encode_target(self):
        """Encoder la variable cible"""
        self.df['DefaultFlag'] = (self.df['Risk'] == 'bad').astype(int)
        logger.info(f"Default rate: {self.df['DefaultFlag'].mean():.2%}")
    
    def _encode_categorical(self):
        """Encoder les variables catégorielles"""
        # Housing
        housing_map = {'own': 1, 'rent': 2, 'free': 3}
        self.df['HousingNum'] = self.df['Housing'].map(housing_map)
        
        # Saving accounts
        saving_map = {'NA': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}
        self.df['SavingAccountsNum'] = self.df['SavingAccounts'].map(saving_map).fillna(0)
        
        # Checking account
        checking_map = {'NA': 0, 'little': 1, 'moderate': 2, 'rich': 3}
        self.df['CheckingAccountNum'] = self.df['CheckingAccount'].map(checking_map).fillna(0)
        
        # Sex
        self.df['SexNum'] = (self.df['Sex'] == 'male').astype(int)
    
    def _create_ratios(self):
        """Créer des ratios et transformations"""
        self.df['LoanToIncome'] = self.df['CreditAmount'] / (self.df['CreditAmount'].median())
        self.df['InstallmentRate'] = self.df['CreditAmount'] / self.df['Duration'] / 100
        self.df['LogCreditAmount'] = np.log1p(self.df['CreditAmount'])
        
        # Time on book (simulé)
        np.random.seed(42)
        self.df['TimeOnBook'] = np.random.randint(1, 60, len(self.df))
        self.df['RemainingTerm'] = np.maximum(0, self.df['Duration'] - self.df['TimeOnBook'] / 12)
    
    def _handle_missing_values(self):
        """Gérer les valeurs manquantes"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna('Unknown')