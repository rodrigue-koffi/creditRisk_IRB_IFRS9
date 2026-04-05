from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """
    Configuration du projet
    """
    # Chemins
    data_path: Path = Path('data')
    output_path: Path = Path('outputs')
    log_path: Path = Path('logs')
    
    # Paramètres modèles
    test_size: float = 0.3
    random_seed: int = 42
    
    # Seuils staging IFRS 9
    relative_threshold: float = 2.0
    absolute_threshold: float = 0.02
    
    # Paramètres macro
    macro_horizon: int = 5
    discount_rate: float = 0.03
    
    # Paramètres Bâle
    correlation_rho: float = 0.12
    confidence_level: float = 0.999
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)