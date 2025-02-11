import numpy as np

def load_dataset(filename):
    data = np.load(filename)
    x_data = data['x']
    y_data = data['y']

    # Organizza i dati in una lista di dizionari
    dataset = [{"x": x_data[:, i], "y": y_data[i]} for i in range(x_data.shape[1])]
    return dataset, x_data.shape[0]


def normalize_data(dataset):
    """
    Normalizza le feature del dataset. 
    Assumiamo che ogni elemento di 'dataset' sia un dizionario con chiave "x" contenente un array NumPy.
    
    Restituisce:
        - Il dataset normalizzato (lista di dizionari con "x" normalizzato)
        - La media delle feature (array NumPy)
        - La deviazione standard delle feature (array NumPy)
    """
    X = np.array([sample["x"] for sample in dataset])
    
    # Calcola la media e la deviazione standard per ciascuna feature (per ogni colonna)
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0) + 1e-8  # aggiunge un piccolo epsilon per evitare divisione per zero
    
    # Normalizza X
    X_norm = (X - mean_X) / std_X

    Y = np.array([sample["y"] for sample in dataset])
    mean_Y = np.mean(Y)
    std_Y = np.std(Y) + 1e-8
    Y_norm = (Y - mean_Y) / std_Y
    
    # Ricostruisci il dataset mantenendo la struttura originale
    normalized_dataset = [{"x": X_norm[i], "y": Y_norm[i]} for i in range(len(dataset))]
    
    return normalized_dataset, mean_X, mean_Y, std_X, std_Y


def denormalize_formula_str(formula_str, mu_x, sigma_x, mu_y, sigma_y):
    """
    Trasforma una formula espressa in termini di variabili normalizzate in una formula denormalizzata.
    
    Args:
        formula_str (str): La formula evoluta in termini di x[i] (normalizzati).
                           Ad esempio: "x[0] * cos(x[2])"
        mu_x (array-like): Media per ciascuna feature.
        sigma_x (array-like): Deviazione standard per ciascuna feature.
        mu_y (float): Media del target.
        sigma_y (float): Deviazione standard del target.
    
    Returns:
        str: La formula in termini delle variabili originali.
    """

    formula_denorm = formula_str
    # x[0] -> ((x[0] - mu_x[0]) / sigma_x[0])
    for i in range(len(mu_x)):  
        formula_denorm = formula_denorm.replace(
            f"x[{i}]", f"((x[{i}] * {sigma_x[i]:.4f}) + {mu_x[i]:.4f})"
        )
    
    
    formula_denorm = f"{mu_y:.4f} + {sigma_y:.4f} * ({formula_denorm})"
    
    return formula_denorm




