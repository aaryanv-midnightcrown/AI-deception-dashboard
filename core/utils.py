import numpy as np

def softmax(x):
    """Numerically stable softmax"""
    e = np.exp(x - np.max(x))
    return e / e.sum()

def normalize(x):
    """Normalize vector to unit length"""
    return x / (np.linalg.norm(x) + 1e-8)
