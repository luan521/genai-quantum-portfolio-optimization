import numpy as np
from itertools import permutations
import cvxpy as cp
import numpy as np
import networkx as nx

def opt_sort_matrix_ring(matrix):
    n = matrix.shape[0]

    f_opt = -np.inf
    opt_perm = None
    for perm in permutations(range(n)):
        perm = list(perm)
        f_perm = ((matrix[perm, perm[1:]+[perm[0]]])**2).sum()
        if f_perm > f_opt:
            opt_perm = perm
            f_opt = f_perm
            
    return opt_perm

# Given a matrix C and a graph of allowed entries G, returns the solution [ lambda, X ] for the above SDP.
def compress_matrix(C, G):
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=200, 
                    nanstr='nan', precision=8, suppress=True, 
                    threshold=1000, formatter=None)
    n = np.shape(C)[0]
    
    # Variables
    X = cp.Variable((n, n), symmetric=True)
    lam = cp.Variable()

    # Constraints
    constraints = [
        X - C << lam * np.eye(n),
        X - C >> -lam * np.eye(n),
    ]
    # Linear restrictions
    G_comp = nx.complement(G)
    constraints += [X[i, j] == 0 for (i, j) in G_comp.edges()]

    # Objective
    objective = cp.Minimize(lam)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-9)

    return lam.value, X.value