import numpy as np

def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return 56465.1066 + 5441856.1240 * (
        np.cos((x[1] * 2.8851) - 0.0583) *
        ((x[0] * 2.8854) - 0.0055) *
        np.cos((x[2] * 2.8925) - 0.0242)
    )

def f3(x: np.ndarray) -> np.ndarray: 
    log_term = np.where(x[:, 2] > 0, np.log(x[:, 2]), 0)  # Evita errori di logaritmo
    return (-np.sin(x[:, 0]) - log_term) + (x[:, 1] * (x[:, 2] + -4.32269))


def f4(x: np.ndarray) -> np.ndarray:
    return np.cos(x[1]) * x[0] * x[0]


def f5(x: np.ndarray) -> np.ndarray: 
    return (np.log(x[0]) + np.sin(x[1])) / (-8.83697 + x[1])


def f6(x: np.ndarray) -> np.ndarray: 
    return (np.log(x[1]) - np.log(x[0])) + x[1]



def f7(x: np.ndarray) -> np.ndarray:
    return (x[0] - x[1]) + (5.96773 + x[0])


def f8(x: np.ndarray) -> np.ndarray:
     return (np.log(x[5]) + (np.cos(x[1]) - ((x[4] + np.sin(x[3])) - x[0])) 
        + (np.sin(x[3]) / ((np.exp(x[5]) - ((x[2] + x[3]) + 0.0606587)) + np.log(x[1]))
        )
    )