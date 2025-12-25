"""
Devign Inference Library - Vulnerability Detection for C Code

Usage:
    from devign_infer import VulnerabilityDetector
    
    detector = VulnerabilityDetector(
        model_path="models/best_model.pt",
        vocab_path="models/vocab.json",
        config_path="models/config.json"
    )
    
    result = detector.analyze('''
        int func(char *input) {
            char buffer[10];
            strcpy(buffer, input);
            return 0;
        }
    ''')
    
    print(result)
    # {'vulnerable': True, 'probability': 0.87, 'risk_level': 'HIGH'}
"""

from .detector import VulnerabilityDetector
from .sarif import SARIFReporter
from .config import InferenceConfig, find_model_path, find_vocab_path, validate_paths

__version__ = "1.0.0"
__all__ = [
    "VulnerabilityDetector",
    "SARIFReporter", 
    "InferenceConfig",
    "find_model_path",
    "find_vocab_path",
    "validate_paths"
]
