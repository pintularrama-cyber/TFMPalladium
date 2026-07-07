import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocesadorProduccion(BaseEstimator, TransformerMixin):
    """Stub para compatibilidad con modelos guardados en el notebook original."""
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class TargetEncoderSmoothing(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smoothing=10):
        self.columns = columns
        self.smoothing = smoothing
        self.mappings = {}
        self.global_mean = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Pass-through: no convertir tipo para no romper la cadena de pasos del pipeline
        return X


class BinaryEncoderCustom(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
