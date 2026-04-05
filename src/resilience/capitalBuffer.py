import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CapitalBuffer:
    """
    Gestion des buffers de capital réglementaire
    """
    
    def __init__(self, rwa: float, cet1_capital: float, ecl: float):
        self.rwa = rwa
        self.cet1_capital = cet1_capital
        self.ecl = ecl
        
    def calculate_required_buffers(self) -> dict:
        """
        Calculer les buffers requis selon Bâle III
        """
        # Buffers réglementaires
        buffers = {
            'CapitalConservationBuffer': self.rwa * 0.025,
            'CountercyclicalBuffer': self.rwa * 0.025,
            'SystemicRiskBuffer': self.rwa * 0.01,
            'TotalBuffer': self.rwa * 0.06
        }
        
        # Capital CET1 requis total
        required_cet1 = self.rwa * 0.045 + buffers['TotalBuffer']
        
        logger.info(f"Buffer total requis: {buffers['TotalBuffer']:,.0f}")
        
        return buffers
    
    def calculate_shortfall(self, el_basel: float) -> float:
        """
        Calculer le shortfall (EL_Bâle - Provisions)
        """
        shortfall = max(0, el_basel - self.ecl)
        
        if shortfall > 0:
            logger.warning(f"Shortfall détecté: {shortfall:,.0f}")
        else:
            logger.info(f"Excédent de provisions: {-shortfall:,.0f}")
        
        return shortfall
    
    def calculate_adjusted_capital(self, shortfall: float) -> dict:
        """
        Calculer le capital ajusté après shortfall
        """
        adjusted_capital = {
            'CET1_Before': self.cet1_capital,
            'CET1_After': max(0, self.cet1_capital - shortfall),
            'Reduction': shortfall,
            'Reduction_Percentage': shortfall / self.cet1_capital if self.cet1_capital > 0 else 0
        }
        
        adjusted_capital['CET1_Ratio_After'] = adjusted_capital['CET1_After'] / self.rwa
        
        return adjusted_capital