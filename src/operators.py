import numpy as np

__all__ = ['BINARY_OPERATORS', 'UNARY_OPERATORS']

# OPERAZIONI BINARIE (due operandi)
def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mul(x, y):
    return x * y

def div(x, y):
    #se y è zero return 1
    y_safe = np.where(np.abs(y) < 1e-10, 1, y)
    return x / y_safe
"""
def div(x, y):
    
    y_safe = np.where(np.abs(y) < 1e-10, 1e-6, y)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = x / y_safe
    return result
"""
"""
def power(x, y):
    # Con NumPy, usa np.where per gestire elementi problematici
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_safe = np.where((x == 0) & (y < 0), 1e-6, x)
    return np.power(x_safe, y)
"""

def power(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    # Se la base è negativa e l'esponente non è un intero -> np.nan
    if np.any((x < 0) & (~np.isclose(y, np.round(y)))):
        return np.nan
    x_safe = np.where((x == 0) & (y < 0), 1e-6, x)
    y_clipped = np.clip(y, -10, 10)
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.power(x_safe, y_clipped)
    return result


# OPERAZIONI UNARIE (un solo operando)
def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def exp(x):
    x_clipped = np.clip(x, -50, 50)
    return np.exp(x_clipped)

def log(x):
    # Evitiamo log di valori non positivi
    x_safe = np.where(x <= 0, 1e-6, x)
    return np.log(x_safe)

# Insiemi di operatori: ogni elemento è una tupla (simbolo, funzione)
BINARY_OPERATORS = [
    ("+", add),
    ("-", sub),
    ("*", mul),
    ("/", div),
    ("^", power)
]

UNARY_OPERATORS = [
    ("sin", sin),
    ("cos", cos),
    ("exp", exp),
    ("log", log)
]

# Variabili disponibili. Per semplicità, consideriamo che ogni campione sia un vettore che viene passato
# come parametro con chiave "x".
VARIABLES = ["x"]

