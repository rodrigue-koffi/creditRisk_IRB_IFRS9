import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Helpers:
    """
    Fonctions utilitaires pour le projet
    """
    
    @staticmethod
    def save_results(results: dict, file_path: str):
        """
        Sauvegarder les résultats au format JSON
        """
        # Convertir les types numpy en types Python
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        serializable_results = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Résultats sauvegardés dans {file_path}")
    
    @staticmethod
    def load_results(file_path: str) -> dict:
        """
        Charger les résultats depuis un fichier JSON
        """
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Résultats chargés depuis {file_path}")
        return results
    
    @staticmethod
    def calculate_weighted_average(values: list, weights: list) -> float:
        """
        Calculer une moyenne pondérée
        """
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)
    
    @staticmethod
    def create_output_directory(base_path: str = 'outputs') -> Path:
        """
        Créer un répertoire de sortie avec timestamp
        """
        from datetime import datetime
        
        output_dir = Path(base_path) / datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Répertoire de sortie créé: {output_dir}")
        return output_dir