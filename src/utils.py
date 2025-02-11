import inspect
from typing import Callable

__all__ = ['arity', 'contains_all_variables', 'variable_counts', 'has_exactly_variables', 'tree_depth']

def arity(f: Callable) -> int:
    """Return the number of expected parameter or None if variable"""
    if inspect.getfullargspec(f).varargs is not None:
        return None
    else:
        return len(inspect.getfullargspec(f).args)
    
def contains_all_variables(individual, num_variables): # required_vars=['x[0]', 'x[1]']):
    """
    Verifica se l'albero (individual) contiene tutte le variabili specificate.
    """
    used = set()
    def traverse(node):
        if node.is_leaf:
            if node.short_name.startswith('x['):  # Assicurati che sia una variabile
                used.add(node.short_name)
        else:
            for child in node._successors:
                traverse(child)
    traverse(individual)
    expected_variables = {f'x[{i}]' for i in range(num_variables)}
    return used >= expected_variables
   
def variable_counts(individual, num_variables):
    """
    Restituisce un dizionario con il conteggio delle occorrenze di ciascuna variabile.
    Le variabili sono attese nel formato "x[0]", "x[1]", ..., "x[num_variables-1]".
    """
    counts = {f'x[{i}]': 0 for i in range(num_variables)}
    
    def traverse(node):
        if node.is_leaf:
            # Se il nodo è una foglia e il suo short_name corrisponde a una variabile attesa, incrementa il conteggio.
            if node.short_name in counts:
                counts[node.short_name] += 1
        else:
            for child in node._successors:
                traverse(child)
    
    traverse(individual)
    return counts



def has_exactly_variables(individual, num_variables):
    """
    Verifica se l'albero 'individual' contiene esattamente una occorrenza per ciascuna variabile attesa.
    Le variabili sono attese nel formato "x[0]", "x[1]", ..., "x[num_variables-1]".
    """
    counts = {f'x[{i}]': 0 for i in range(num_variables)}
    
    def traverse(node):
        if node.is_leaf:
            # Se il nodo è una foglia e il suo short_name corrisponde a una variabile attesa,
            # incrementa il conteggio.
            if node.short_name in counts:
                counts[node.short_name] += 1
        else:
            for child in node._successors:
                traverse(child)
    
    traverse(individual)
    # Restituisce True solo se ogni variabile appare esattamente una volta
    return all(count >= 1 for count in counts.values())

def pretty_print(node):
    """
    Converte ricorsivamente l'albero (node) in una stringa leggibile in notazione matematica.
    """
    # Se il nodo è una foglia, restituisci il suo nome (variabile o costante)
    if node.is_leaf:
        return str(node.short_name)
    
    # Se il nodo ha un solo figlio, è un operatore unario
    if len(node._successors) == 1:
        child_str = pretty_print(node._successors[0])
        return f"{node.short_name}({child_str})"
    
    # Se il nodo ha due figli, assumiamo che sia un operatore binario
    if len(node._successors) == 2:
        left_str = pretty_print(node._successors[0])
        right_str = pretty_print(node._successors[1])
        return f"({left_str} {node.short_name} {right_str})"
    
    # Per operatori con più di due figli, li uniamo separando con una virgola
    children_str = [pretty_print(child) for child in node._successors]
    return f"{node.short_name}(" + ", ".join(children_str) + ")"

def tree_depth(tree):
    """Calcola la profondità di un albero."""
    if not hasattr(tree, "_successors") or not tree._successors:
        return 1  # Un nodo foglia ha profondità 1
    return 1 + max(tree_depth(child) for child in tree._successors)
