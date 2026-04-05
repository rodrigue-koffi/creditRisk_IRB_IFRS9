import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.orchestration.mainOrchestrator import MainOrchestrator
from src.utils.logger import setup_logger

logger = setup_logger('PipelineExecutor', log_file='logs/execution.log')

class PipelineExecutor:
    """
    Exécuteur du pipeline avec gestion des erreurs
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.orchestrator = None
        
    def execute(self) -> dict:
        """
        Exécuter le pipeline avec gestion des erreurs
        """
        try:
            logger.info("Initialisation de l'orchestrateur")
            self.orchestrator = MainOrchestrator(self.data_path)
            
            logger.info("Exécution du pipeline complet")
            results = self.orchestrator.run_full_pipeline()
            
            logger.info("Pipeline exécuté avec succès")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    executor = PipelineExecutor(data_path='data')
    results = executor.execute()