from node import Node
import numpy as np
import random 
from dag import DagGP
from operators import BINARY_OPERATORS, UNARY_OPERATORS
from utils import *
import copy

"""
def fitness(tree, dataset, num_variables):
    
    Calcola la fitness di un albero rispetto al dataset con miglioramenti.
    
    error = 0
    penalty = 0

    for sample in dataset:
        try:
            pred = tree(x=sample["x"])
            pred = np.clip(pred, -1e3, 1e3)  # Evita valori estremi
            if np.isnan(pred) or np.isinf(pred):
                penalty += 1e6
                continue
        except Exception:
            penalty += 1e8
            continue

        delta = (pred - sample["y"]) ** 2
        if delta > 1e11:
            penalty += 1e6
        error += delta

    dataset_values = np.array([sample["y"] for sample in dataset])
    target_std = np.std(dataset_values)
    mse_value = (error / len(dataset)) / (target_std + 1e-6)

    max_depth = 10
    #depth_penalty = 0.002 * max(0, tree.depth - max_depth)
    #complexity_penalty = 0.002 * len(tree) + depth_penalty
    complexity_penalty = 0.002 * len(tree)
    counts = variable_counts(tree, num_variables)
    var_penalty = sum(3 * max(0, count - 1) for count in counts.values())
    unused_vars = num_variables - len(counts)
    var_penalty += 5 * unused_vars

    return mse_value + complexity_penalty + var_penalty + penalty
"""
"""
def fitness(tree, dataset, num_variables, batch_size=None, early_stop_threshold=1e14):
    
    Calcola la fitness di un albero rispetto al dataset con ottimizzazioni:
    - NumPy per velocizzare i calcoli
    - Early stopping per interrompere valutazioni inutili
    - Batch evaluation per migliorare l'efficienza
    
    dataset_size = len(dataset)
    
    if batch_size is None or batch_size > dataset_size:
        batch_size = dataset_size  # Usa l'intero dataset se batch_size non è definito
    
    indices = np.random.choice(dataset_size, batch_size, replace=False)  # Campiona dati casuali
    
    X_batch = np.array([dataset[i]["x"] for i in indices])  # Estrai le feature
    Y_batch = np.array([dataset[i]["y"] for i in indices])  # Estrai i target

    try:
       preds = np.array([tree(x=x) for x in X_batch])  # Valuta il modello su tutto il batch
       preds = np.clip(preds, -1e3, 1e3)  # Evita valori estremi
    except Exception:
       return 1e8  # Penalità enorme in caso di errore

    #if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
    #    return 1e6  # Penalità per output non validi
    
    errors = (preds - Y_batch) ** 2  # Errore quadratico
    cumulative_error = np.sum(errors)

    # Early stopping: interrompe se il fitness è già troppo alto
    #if cumulative_error > early_stop_threshold:
    #    return 1e15  # Penalità alta

    # Normalizzazione rispetto alla deviazione standard del target
    target_std = np.std(Y_batch) + 1e-6
    mse_value = (cumulative_error / batch_size) / target_std  

    # Penalità sulla complessità dell'albero
    complexity_penalty = 0.002 * len(tree)

    # Penalità sulle variabili usate
    counts = variable_counts(tree, num_variables)
    #var_penalty = sum(3 * max(0, count - 1) for count in counts.values())
    unused_vars = num_variables - len(counts)
    var_penalty = 5 * unused_vars

    return mse_value + complexity_penalty + var_penalty
"""

"""
def fitness(tree, dataset, num_variables, batch_size=None, early_stop_threshold=1e14):
    
    Calcola il Mean Squared Error (MSE) di un albero rispetto al dataset.
    
    Ogni elemento del dataset è un dizionario con:
      - "x": un array NumPy contenente le feature del campione
      - "y": il valore target corrispondente
    
    Se l'albero genera un errore durante la valutazione o restituisce un valore non valido (NaN o inf),
    viene restituito un valore di penalità molto elevato.
    
    Args:
        tree: L'albero (un'istanza della classe Node) da valutare.
        dataset: Una lista di dizionari, dove ogni dizionario contiene "x" e "y".
        
    Returns:
        mse: Il Mean Squared Error calcolato sui campioni del dataset.
    
    total_error = 0.0
    n_samples = len(dataset)
    
    #for sample in dataset:
    
         # Valutiamo l'albero per ciascun campione
     #   pred = tree(x=sample["x"])
      #  if not np.isfinite(pred):
       #     return 1e12

        # Calcoliamo l'errore quadratico
        #total_error += (pred - sample["y"]) ** 2
    
    #mse = total_error / n_samples


    for  sample in dataset:
        pred = tree(x=sample["x"])
        y_true = sample["y"]

        if not np.isfinite(pred):
            total_error += 1e6  # Penalità solo per quel campione, non per tutta la generazione

        if y_true == 0:  
            continue  # Salta i campioni con y=0 per evitare divisioni per zero

        error = abs((pred - y_true) / y_true)  # Errore percentuale assoluto
        total_error += error
    mape = total_error / n_samples
    
    return mape
"""


def fitness(tree, dataset, num_variables, clip_value=None, batch_size=None, early_stop_threshold=1e14):
    """
    Calcola il Mean Squared Error (MSE) in modo safe per un albero rispetto a un dataset.
    
    Ogni elemento del dataset è un dizionario con:
      - "x": un array NumPy contenente le feature del campione
      - "y": il valore target corrispondente

    Viene applicato un clipping a predizioni e target per evitare valori troppo estremi che
    potrebbero causare overflow o underflow.

    Args:
        tree: L'albero (un'istanza della classe Node) da valutare.
        dataset: Una lista di dizionari, dove ogni dizionario contiene "x" e "y".
        clip_value: Valore massimo (in valore assoluto) oltre il quale clipperemo predizioni e target.
                    Se None, viene calcolato come 100 * mediana(|y|) (oppure un valore di default se la mediana è 0).

    Returns:
        mse: Il Mean Squared Error calcolato sul dataset.
    """
    n_samples = len(dataset)
    
    # Se non viene fornito un clip_value, lo calcoliamo dai target
    if clip_value is None:
        targets = np.array([sample["y"] for sample in dataset])
        median_target = np.median(np.abs(targets))
        clip_value = median_target * 1 if median_target != 0 else 1e6

    total_error = 0.0
    
    for sample in dataset:
        try:
            # Valutiamo l'albero per il campione corrente
            pred = tree(x=sample["x"])
            # Applichiamo il clipping alle predizioni e al target
            pred = np.clip(pred, -clip_value, clip_value)
            target = np.clip(sample["y"], -clip_value, clip_value)
        except Exception:
            # Se c'è un errore durante la valutazione, penalizzo
            return 1e12
        
        # Se la predizione non è finita, penalità elevata
        if not np.isfinite(pred):
            return 1e12
        
        # Calcoliamo l'errore quadratico per il campione corrente
        error = (pred - target) ** 2
        total_error += error

    mse = total_error / n_samples

    counts = variable_counts(tree, num_variables)
    unused_vars = sum(1 for v in counts.values() if v == 0)  # Conta solo le variabili mai usate
    var_penalty = mse * 10 * unused_vars  # Penalità più significativa
    return mse + var_penalty



def tournament_selection(population, dataset, num_variables, batch_size, k=5):
    """
    Seleziona il miglior individuo da un sottoinsieme casuale della popolazione.
    
    Args:
        population: Lista degli alberi (individui).
        dataset: Il dataset usato per calcolare la fitness.
        k: Numero di individui selezionati casualmente.

    Returns:
        Il miglior albero (con fitness minore).
    """
    selected = random.sample(population, k)  # Prende k individui casuali
    selected.sort(key=lambda t: fitness(t, dataset, num_variables, batch_size=batch_size))  # Ordina in base alla fitness
    return selected[0]  # Ritorna il migliore


""" 
def mutate(tree, dag, num_variables, prob=0.2, n_nodes=7):
    
    if random.random() < prob:
        return dag.create_individual(num_variables, n_nodes)  # Sostituisce completamente con un nuovo albero
    #print(f"mutate {tree.long_name}")
    
    if not tree._successors:
        return tree

    new_successors = [mutate(child, dag, num_variables, prob, n_nodes - 1) for child in tree._successors]
    tree._successors = new_successors
    return tree
"""  

def mutate(tree, dag, num_variables, prob, best_fitness, prev_fitness, n_nodes=7, generation=1):
    """
    Esegue una mutazione adattiva su un albero, scegliendo il tipo di mutazione in base al contesto.

    Args:
        tree: L'albero da mutare.
        dag: Il DAG per generare nuovi individui.
        num_variables: Numero di variabili disponibili.
        prob: Probabilità di mutare un nodo.
        n_nodes: Numero massimo di nodi del nuovo sottoalbero.
        generation: Numero della generazione attuale.
        best_fitness: Miglior fitness globale (opzionale, per adattare le mutazioni).
        prev_fitness: Fitness dell'albero nelle generazioni precedenti.

    Returns:
        L'albero mutato.
    """  
    # Se la mutazione deve avvenire con probabilità prob
    if random.random() < prob:
        return dag.create_individual(num_variables, n_nodes)  # Sostituisce completamente con un nuovo albero

    # Controlliamo la profondità dell'albero
    depth = tree_depth(tree)

    # **Adattiamo le probabilità delle mutazioni**
    aggressive = best_fitness and prev_fitness and prev_fitness - best_fitness < 0.01
    
    mutation_weights = {
        "replace_subtree": 0.4 if aggressive else 0.2, 
        "change_operator": 0.05 if depth > 5 else 0.2, 
        "change_constant": 0.25,  
        "swap_subtrees": 0.1 if depth > 5 else 0.2, 
        "structural_light": 0.2, 
    }

    # Scegliamo il tipo di mutazione in base ai pesi adattivi
    mutation_type = random.choices(
        list(mutation_weights.keys()),
        weights=list(mutation_weights.values())
    )[0]

    # **Eseguiamo la mutazione scelta**
    if mutation_type == "replace_subtree":
        return dag.create_individual(num_variables, n_nodes)
    
    elif mutation_type == "change_operator":
        return tree.change_operator()

    elif mutation_type == "change_constant":
        return tree.change_constant()

    elif mutation_type == "swap_subtrees":
        return tree.swap_subtrees()

    elif mutation_type == "structural_light":
        return tree.structural_light(dag, num_variables, n_nodes)

    # Se non possiamo mutare, restituiamo l'albero originale
    return tree


"""

def crossover(tree1, tree2):
    
    Esegue il crossover tra due alberi scambiando sottoalberi casualmente.

    Args:
        tree1: Il primo albero genitore.
        tree2: Il secondo albero genitore.

    Returns:
        Un nuovo albero figlio.
    
    if random.random() < 0.2:
        return tree1  # A volte restituisce uno dei genitori senza cambiamenti
    
    if not tree1._successors or not tree2._successors:
        return tree1  # Se uno dei due è una foglia, non si può fare crossover

    # Scegliamo casualmente un figlio da scambiare
    idx1 = random.randint(0, len(tree1._successors) - 1)
    idx2 = random.randint(0, len(tree2._successors) - 1)

    # Creiamo una copia dei figli per non modificare gli originali
    new_successors1 = list(tree1._successors)
    new_successors2 = list(tree2._successors)

    # Scambiamo i sottoalberi
    new_successors1[idx1], new_successors2[idx2] = new_successors2[idx2], new_successors1[idx1]

    # Creiamo nuovi alberi con i figli scambiati
    new_tree1 = Node(tree1._func, new_successors1, name=tree1.short_name)
    return new_tree1
"""

def crossover(tree1, tree2):
    """
    Esegue il crossover tra due alberi scambiando sottoalberi casualmente.

    Args:
        tree1: Il primo albero genitore.
        tree2: Il secondo albero genitore.

    Returns:
        Due nuovi alberi figli.
    """
    if random.random() < 0.2:
        return tree1, tree2  # A volte restituisce i genitori senza cambiamenti

    if not tree1._successors or not tree2._successors:
        return tree1, tree2  # Se uno dei due è una foglia, non si può fare crossover

    # Scegliamo casualmente un figlio da scambiare
    idx1 = random.randint(0, len(tree1._successors) - 1)
    idx2 = random.randint(0, len(tree2._successors) - 1)

    # Creiamo una copia dei figli per non modificare gli originali
    new_successors1 = list(tree1._successors)
    new_successors2 = list(tree2._successors)

    # Scambiamo i sottoalberi
    new_successors1[idx1], new_successors2[idx2] = new_successors2[idx2], new_successors1[idx1]

    # Creiamo nuovi alberi con i figli scambiati
    new_tree1 = Node(tree1._func, new_successors1, name=tree1.short_name)
    new_tree2 = Node(tree2._func, new_successors2, name=tree2.short_name)

    return new_tree1, new_tree2



"""
def genetic_programming(dataset, population_size, generations, tree_depth, num_variables, batch_size):
    
    Esegue l'algoritmo di Genetic Programming per trovare la migliore formula simbolica.

    Args:
        dataset: Il dataset usato per calcolare la fitness.
        population_size: Numero di individui nella popolazione.
        generations: Numero di generazioni da eseguire.
        tree_depth: Profondità massima degli alberi.

    Returns:
        Il miglior albero trovato.
    
    dag = DagGP(BINARY_OPERATORS + UNARY_OPERATORS, num_variables, constants=5)

    elite_size = max(1, population_size // 5)
    # Generazione della popolazione iniziale
    population = [dag.create_individual(num_variables=num_variables) for _ in range(population_size)]
    best = None

    for gen in range(generations):
        # Ordina la popolazione in base alla fitness (minore errore è migliore)
        population.sort(key=lambda t: fitness(t, dataset, num_variables, batch_size=batch_size))
        current_best = population[0]
        current_fitness = fitness(current_best, dataset, num_variables, batch_size=batch_size)

        print(f"Generazione {gen}: Best fitness = {current_fitness:.4f}, Formula = {current_best.long_name}")

        # Salviamo il miglior individuo trovato finora
        if best is None or current_fitness < fitness(best, dataset, num_variables, batch_size=batch_size):
            best = current_best

        # Se troviamo un errore molto basso, interrompiamo l'evoluzione
        if current_fitness < 1e-6:
            break

        # Creazione della nuova popolazione tramite selezione, crossover e mutazione
        #new_population = population[:population_size // 2]  # Manteniamo i migliori individui
        new_population = population[:elite_size]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, dataset, num_variables, k=3)
            parent2 = tournament_selection(population, dataset, num_variables, k=3)
            child = crossover(parent1, parent2)
            prev_fitness = fitness(child, dataset, num_variables)
            child = mutate(child, dag, num_variables, prob=0.2, n_nodes=tree_depth, prev_fitness=prev_fitness)
            new_population.append(child)

        population = new_population  # Aggiorniamo la popolazione

    return best  # Restituisce il miglior albero trovato
"""

def genetic_programming(dataset, population_size, generations, tree_depth, num_variables, batch_size):
    """
    Esegue l'algoritmo di Genetic Programming per trovare la migliore formula simbolica.

    Args:
        dataset: Il dataset usato per calcolare la fitness.
        population_size: Numero di individui nella popolazione.
        generations: Numero di generazioni da eseguire.
        tree_depth: Profondità massima degli alberi.

    Returns:
        Il miglior albero trovato.
    """
    dag = DagGP(
        BINARY_OPERATORS + UNARY_OPERATORS, 
        num_variables, 
        constants=np.random.uniform(-10, 10, size=5)  # Genera 5 costanti casuali
    )

    elite_size = max(1, population_size // 5)
    # Generazione della popolazione iniziale
    population = [dag.create_individual(num_variables=num_variables) for _ in range(population_size)]
    #while any(np.isnan(fitness(ind, dataset, num_variables, batch_size)) for ind in population):
     #   population = [dag.create_individual(num_variables=num_variables) for _ in range(population_size)]
    best = None
    stagnation_counter = 0
    prev_best_fitness = None

    for gen in range(generations):
        # Ordina la popolazione in base alla fitness
        population.sort(key=lambda t: fitness(t, dataset, num_variables, batch_size=batch_size))
        current_best = population[0]
        current_fitness = fitness(current_best, dataset, num_variables, batch_size=batch_size)

        print(f"Generazione {gen}: Best fitness = {current_fitness:.4f}, Formula = {current_best.long_name}")

        # Se il miglior individuo non migliora, incrementa il contatore di stagnazione
        if prev_best_fitness is not None and abs(current_fitness - prev_best_fitness) < 1e-6:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        prev_best_fitness = current_fitness

        # Se la fitness è alta e la stagnazione dura, aumenta il tasso di mutazione
        if stagnation_counter >= 3:
            mutation_prob = 0.7  # Aumenta significativamente la probabilità di mutazione
            print("stagnation: aumento mutation prob")
        else:
            mutation_prob = 0.3

        # Se la fitness è sufficientemente bassa, interrompi l'evoluzione
        if current_fitness < 1e-10:
            break

        # Creazione della nuova popolazione
        new_population = population[:elite_size]
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, dataset, num_variables, batch_size, k=3)
            parent2 = tournament_selection(population, dataset, num_variables, batch_size, k=3)
            #child = crossover(parent1, parent2)
            child, child2 = crossover(parent1, parent2)
            prev_fitness = fitness(child, dataset, num_variables, batch_size=batch_size)
            prev_fitness2 = fitness(child2, dataset, num_variables, batch_size=batch_size)
            child = mutate(child, dag, num_variables, mutation_prob, current_fitness, prev_fitness, n_nodes=tree_depth)
            child2 = mutate(child2, dag, num_variables, mutation_prob, current_fitness, prev_fitness2, n_nodes=tree_depth)
            new_population.append(child)
            if len(new_population) < population_size:
                new_population.append(child2)
        population = new_population

    return current_best  # Restituisce il miglior albero trovato

