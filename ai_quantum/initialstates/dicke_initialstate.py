import numpy as np
from math import comb
from qiskit import QuantumCircuit

def dicke_statevector(n_assets, B):
    """
    Args:
        n_assets (int): Number of assets.
        B (float): Budget parameter.
        
    Returns:
        (qiskit.circuit.quantumcircuit.QuantumCircuit): Quantum circuit initialized with the statevector for the n_assets-qubit Dicke state of weight B.
    """
    dim = 2**n_assets
    state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        if bin(i).count('1') == B:
            state[i] = 1.0
    state /= np.sqrt(comb(n_assets, B))
    
    qc = QuantumCircuit(n_assets, n_assets)
    qc.initialize(state, range(n_assets))
    return qc
