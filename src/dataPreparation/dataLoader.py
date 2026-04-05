import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Chargeur de données pour le projet de risque de crédit
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_data = None
        self.macro_data = None
        
    def load_german_credit_data(self) -> pd.DataFrame:
        """
        Charger les données German Credit Data
        """
        try:
            file_path = self.data_path / 'german_credit_data.xlsx'
            self.raw_data = pd.read_excel(file_path, sheet_name='german_credit_data(1)')
            logger.info(f"Données chargées : {self.raw_data.shape[0]} lignes, {self.raw_data.shape[1]} colonnes")
            return self.raw_data
        except Exception as e:
            logger.error(f"Erreur lors du chargement : {e}")
            raise
    
    def load_macro_data(self, file_name: str = 'macro_data.csv') -> pd.DataFrame:
        """
        Charger les données macroéconomiques
        """
        try:
            file_path = self.data_path / file_name
            if file_path.exists():
                self.macro_data = pd.read_csv(file_path)
            else:
                # Créer des données macro par défaut
                self.macro_data = self._create_default_macro_data()
            logger.info(f"Données macro chargées : {self.macro_data.shape}")
            return self.macro_data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données macro : {e}")
            raise
    
    def _create_default_macro_data(self) -> pd.DataFrame:
        """
        Créer des données macro par défaut pour l'Allemagne (hors COVID)
        """
        years = np.arange(2000, 2020)
        
        # Taux de chômage allemand
        unemployment = [7.8, 7.6, 8.6, 9.3, 9.8, 10.6, 9.8, 8.7, 7.5, 7.8,
                        7.1, 6.0, 5.5, 5.3, 5.0, 4.6, 4.1, 3.8, 3.4, 3.1]
        
        # Taux d'intérêt (Euribor 3 mois)
        interest_rates = [3.5, 3.3, 3.0, 2.8, 2.5, 2.7, 3.1, 3.8, 4.0, 3.2,
                          2.0, 1.5, 0.8, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
        
        # Croissance PIB
        gdp_growth = [2.9, 2.0, 1.5, 1.0, 0.5, 1.5, 3.0, 2.5, 1.0, -5.0,
                      3.0, 2.5, 1.5, 1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0]
        
        # Indice des prix immobiliers
        hpi = [100, 101, 102, 103, 104, 105, 108, 112, 115, 118,
               120, 125, 130, 135, 140, 148, 155, 162, 170, 178]
        
        return pd.DataFrame({
            'Year': years,
            'UnemploymentRate': unemployment,
            'InterestRate': interest_rates,
            'GdpGrowth': gdp_growth,
            'HpiIndex': hpi
        })