from typing import Callable
import numbers
from utils import arity
import random
from operators import *

__all__ = ['Node']

class Node:
    _func: Callable
    _successors: tuple['Node']
    _arity: int
    _str: str

    def __init__(self, node=None, successors=None, *, name=None):
        if callable(node):
            def _f(*_args, **_kwargs):
                return node(*_args)

            self._func = _f
            self._successors = tuple(successors)
            self._arity = arity(node)
            assert self._arity is None or len(tuple(successors)) == self._arity, (
                "Panic: Incorrect number of children."
                + f" Expected {len(tuple(successors))} found {arity(node)}"
            )
            self._leaf = False
            assert all(isinstance(s, Node) for s in successors), "Panic: Successors must be `Node`"
            self._successors = tuple(successors)
            if name is not None:
                self._str = name
            elif node.__name__ == '<lambda>':
                self._str = 'λ'
            else:
                self._str = node.__name__
        elif isinstance(node, str):
            #self._func = eval(f'lambda **_kw: {node}')
            self._func = eval(f'lambda x, **_kw: x[{node[2:-1]}]')
            self._successors = tuple()
            self._arity = 0
            self._str = node
        elif isinstance(node, (int, float)):
            #self._func = eval(f'lambda *, {node}, **_kw: {node}')
            self._func = eval(f'lambda **_kw: {node}')
            self._successors = tuple()
            self._arity = 0
            #self._str = str(node)
            self._str = f'{node:g}'
        else:
            raise TypeError(f"Tipo non supportato per 'node': {type(node)}. Expected str or number.")
            assert False
        if name is not None:
            self._str = name
        self._long_name = self._str

    def __call__(self, **kwargs):
        return self._func(*[c(**kwargs) for c in self._successors], **kwargs)

    def __str__(self):
        return self.long_name

    def __len__(self):
        return 1 + sum(len(c) for c in self._successors)

    @property
    def value(self):
        return self()

    @property
    def arity(self):
        return self._arity

    @property
    def successors(self):
        return list(self._successors)

    @successors.setter
    def successors(self, new_successors):
        assert len(new_successors) == len(self._successors)
        self._successors = tuple(new_successors)

    @property
    def is_leaf(self):
        return not self._successors

    @property
    def short_name(self):
        return self._str

    @property
    def long_name(self):
        if self.is_leaf:
            return self.short_name
        else:
            return f'{self.short_name}(' + ', '.join(c.long_name for c in self._successors) + ')'

    @property
    def subtree(self):
        result = set()
        _get_subtree(result, self)
        return result

    #def draw(self):
     #   try:
      #      return draw(self)
       # except Exception as msg:
        #    warnings.warn(f"Drawing not available ({msg})", UserWarning, 2)
         #   return None

    def change_operator(self):
        """
        Cambia l'operatore del nodo, se è un nodo operatore (ovvero, non è una foglia).
        Viene scelto un nuovo operatore dello stesso tipo (unario o binario) da un insieme predefinito.
        """
        if self._arity is None:
            return self
        # Verifica che il nodo non sia una foglia (deve avere un'arità > 0)
        if self._arity > 0:
            if self._arity == 1:
                # Se l'arità è 1, scegli tra gli operatori unari
                new_op = random.choice(UNARY_OPERATORS)[1]
            elif self._arity == 2:
                # Se l'arità è 2, scegli tra gli operatori binari
                new_op = random.choice(BINARY_OPERATORS)[1]
            else:
                # Per altri casi, scegli tra tutti gli operatori
                new_op = random.choice(BINARY_OPERATORS + UNARY_OPERATORS)[1]
            # Aggiorna la funzione associata al nodo
            self._func = lambda *args, **kwargs: new_op(*args)
            # Aggiorna la rappresentazione testuale
            self._str = new_op.__name__
        return self

    def change_constant(self):
        """
        Se il nodo rappresenta una costante(arity == 0),
        modifica leggermente il suo valore aggiungendo una piccola perturbazione casuale.
        """
        try:
            # Prova a interpretare il nome come un numero
            val = float(self._str)
        except Exception:
            return self  # Se non è possibile, ritorna il nodo invariato
        delta = random.uniform(-0.5, 0.5)
        new_val = val + delta
        self._func = eval(f'lambda **_kw: {new_val}')
        self._str = f'{new_val:g}'
        return self

    def swap_subtrees(self):
        """
        Se il nodo ha almeno due successori, scambia casualmente due di essi.
        """
        if not self._successors or len(self._successors) < 2:
            return self
        # Converti i successori in una lista
        new_successors = list(self._successors)
        i, j = random.sample(range(len(new_successors)), 2)
        new_successors[i], new_successors[j] = new_successors[j], new_successors[i]
        self._successors = tuple(new_successors)
        return self

    def structural_light(self, dag, num_variables, n_nodes=3):
        """
        Esegue una leggera modifica strutturale: sostituisce un sottoalbero scelto casualmente
        con un nuovo sottoalbero generato dal DAG.
        """
        if self._successors:
            new_successors = list(self._successors)
            idx = random.randint(0, len(new_successors) - 1)
            # Genera un nuovo sottoalbero con una profondità più ridotta
            new_subtree = dag.create_individual(num_variables, n_nodes)
            new_successors[idx] = new_subtree
            self._successors = tuple(new_successors)
        return self


def _get_subtree(bunch: set, node: Node):
    bunch.add(node)
    for c in node._successors:
        _get_subtree(bunch, c)

