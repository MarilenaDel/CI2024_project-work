from typing import Collection
from node import Node
import random
from utils import *

__all__ = ['DagGP']


class DagGP:
    def __init__(self, operators: Collection, variables: int | Collection, constants: int | Collection):
        self._operators = list(operators)
        if isinstance(variables, int):
            self._variables = [Node(f"x[{i}]") for i in range(variables)]
        else:
            self._variables = [Node(t) for t in variables]
        print("Variabili:", [v.long_name for v in self._variables])  # Debug
        if isinstance(constants, int):
            print("Generazione costanti:", [random.uniform(-5, 5) for _ in range(constants)])
            self._constants = [Node(random.uniform(-5, 5)) for i in range(constants)]
        else:
            self._constants = [Node(t) for t in constants]
        print("Costanti:", [c.long_name for c in self._constants])  # Debug delle costanti

    def create_individual(self, num_variables, n_nodes=7):
        pool = self._variables * (1 + len(self._constants) // len(self._variables)) + self._constants
        #variables_used = set()
        individual = None
        candidate = None
        max_attempts = 1000
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            candidate = None

            while candidate is None or len(candidate) < n_nodes:
                op_sym, op_func = random.choice(self._operators)
                #params = random.choices(pool, k=arity(op_func))
                #candidate = Node(op_func, params, name=op_sym)
                #pool.append(candidate)
                arity_value = arity(op_func)

                if arity_value == 1:  # Se è un operatore unario
                    param = random.choice(self._variables)  # Solo variabili, NO costanti
                    candidate = Node(op_func, [param], name=op_sym)

                elif arity_value == 2:  # Se è un operatore binario
                    param1 = random.choice(pool)  # Variabili o costanti
                    param2 = random.choice(pool)
                    candidate = Node(op_func, [param1, param2], name=op_sym)

                pool.append(candidate)
                #print(f"Tentativo {attempts}: {candidate}")

                #if has_exactly_variables(candidate, num_variables):
                if contains_all_variables(candidate, num_variables):
                    return candidate
        raise RuntimeError("100 tentativi")
                
                #if contains_all_variables(candidate, num_variables):
                 #   return candidate
        #while individual is None or len(individual) < n_nodes:
         #   op_sym, op_func = random.choice(self._operators)  # Estrai il simbolo e la funzione
            
          #  params = random.choices(pool, k=arity(op_func))
           # for param in params:
            #    if isinstance(param, Node) and param.short_name in ['x[0]', 'x[1]']:
             #       variables_used.add(param.short_name)

            #individual = Node(op_func, params, name=op_sym)
            #pool.append(individual)
            #if len(variables_used) == 2:
             #   break
        #return individual

    @staticmethod
    def default_variable(i: int) -> str:
        return f'x{i}'

    @staticmethod
    def evaluate(individual: Node, X, variable_names=None):
        if variable_names:
            names = variable_names
        else:
            names = [DagGP.default_variable(i) for i in range(len(X[0]))]

        y_pred = list()
        for row in X:
            y_pred.append(individual(**{n: v for n, v in zip(names, row)}))
        return y_pred

    @staticmethod
    def plot_evaluate(individual: Node, X, variable_names=None):
        import matplotlib.pyplot as plt

        y_pred = DagGP.evaluate(individual, X, variable_names)
        plt.figure()
        plt.title(individual.long_name)
        plt.scatter([x[0] for x in X], y_pred)

        return y_pred

    @staticmethod
    def mse(individual: Node, X, y, variable_names=None):
        y_pred = DagGP.evaluate(individual, X, variable_names)
        return sum((a - b) ** 2 for a, b in zip(y, y_pred)) / len(y)