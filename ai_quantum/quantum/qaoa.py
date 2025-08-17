from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
import numpy as np

class QAOA():
    """
    Implements QAOA for portfolio optimization using a quantum circuit to solve QUBO problems.
    
    Attributes:
        expected_value (list[float]): List of asset returns.
        cov_matrix (DataFrame-like): Covariance matrix.
        q (float): Covariance scaling factor.
        B (float): Budget/threshold parameter.
        lamb (float): Penalty parameter.
        qc (qiskit.circuit.quantumcircuit.QuantumCircuit): Quantum circuit already initialized.
        mixture_layer (str): x, ring_mixer.
        edges_hc (list): list of tuples containing all connected qubits in the cost hamiltonian
        edges_hb (list): list of tuples containing all connected qubits in the mixture hamiltonian
    """
    
    def __init__(self, expected_value, cov_matrix, q, B, lamb, qc=None, mixture_layer='x', edges_hc=None, edges_hb=None):
        """
        Initializes the QAOA instance and prepares the quantum circuit.
        
        Args:
            expected_value (list[float]): List of asset returns.
            cov_matrix (DataFrame-like): Covariance matrix.
            q (float): Covariance scaling factor.
            B (float): Budget/threshold parameter.
            lamb (float): Penalty parameter.
            qc (qiskit.circuit.quantumcircuit.QuantumCircuit): Quantum circuit already initialized.
            mixture_layer (str): x, ring_mixer.
            edges_hc (list): list of tuples containing all connected qubits in the cost hamiltonian
            edges_hb (list): list of tuples containing all connected qubits in the mixture hamiltonian
        """
        
        self.q = q
        self.B = B
        self.lamb = lamb
        self.expected_value = expected_value
        self.cov_matrix = cov_matrix
        self.n_assets = len(expected_value)
        self.mixture_layer = mixture_layer
        
        if qc is not None:
            self.qc0 = qc.copy()
        else:
            self.qc0 = QuantumCircuit(self.n_assets)
            # Initialization - prepare an equal superposition state
            for qubit in range(self.n_assets):
                self.qc0.h(qubit)
            self.qc0.barrier()
            
        self.qc = self.qc0.copy()
        
        self.edges_complete = [(i, j) for i in range(self.n_assets) for j in range(i+1, self.n_assets)]
            
        if edges_hc is None:
            self.edges_hc = self.edges_complete
        else:
            self.edges_hc = edges_hc
            
        if edges_hb is None:
            self.edges_hb = self.edges_complete
        else:
            self.edges_hb = edges_hb
            
    def restart(self):
        """
        Restart the circuit
        """
        
        self.qc = self.qc0.copy()
    
    def cost_hamiltonian_wheight(self, i, j=None):
        """
        Calculate the weights for the Hamiltonian of the QUBO problem.
        
        Args:
            i (int): Index of the first asset.
            j (int, optional): Index of the second asset. Defaults to None.
            
        Returns:
            float: The weight that multiplies the product of the Z operators, acting on qubits i and j.
        """
        
        if j is None:
            response =self.expected_value[i]+self.lamb*(2*self.B-self.n_assets)-self.q*self.cov_matrix[i].sum()
        else:
            response = self.q*self.cov_matrix[i][j]+self.lamb
        return response
    
    def draw(self):
        """
        Draws the quantum circuit using matplotlib.
        """
        self.qc.draw(output="mpl", style="iqp")
        
    def _add_mixture_layer(self, beta):
        """
        Implement exp(-i*beta*H_B).
        
        Args:
            beta (float): Parameter for the mixing Hamiltonian.
        """
        
        if self.mixture_layer == 'x':
            for qubit in range(self.n_assets):
                self.qc.rx(2*beta, qubit)
        elif self.mixture_layer == 'ring_mixer':
            for e in self.edges_hb:
                self.qc.rxx(
                            beta, 
                            e[0], 
                            e[1]
                           )
                self.qc.ryy(
                            beta, 
                            e[0], 
                            e[1]
                           )
    
    def add_layer(self, gamma, beta):
        """
        Adds one QAOA layer to the circuit.
        
        This layer applies:
            - Cost Hamiltonian: exp(-i*gamma*H_c) using CNOT and RZ gates.
            - Mixing Hamiltonian: exp(-i*beta*H_B) using RX gates.
        
        Args:
            gamma (float): Parameter for the cost Hamiltonian.
            beta (float): Parameter for the mixing Hamiltonian.
        """
        
        # Implement exp(-i*gamma*H_c)
        # H_c: Cost Hamiltonian
        for e in self.edges_hc:
            self.qc.cx(e[0], e[1])
            self.qc.rz(2*gamma*self.cost_hamiltonian_wheight(e[0], e[1]), e[1])
            self.qc.cx(e[0], e[1])
        for qubit in range(self.n_assets):
            self.qc.rz(2*gamma*self.cost_hamiltonian_wheight(qubit), qubit)
        self.qc.barrier()
                
        # Implement exp(-i*beta*H_B)
        self._add_mixture_layer(beta)
        self.qc.barrier()
             
    def measure_energy(self, precision=1e-3):
        """
        Measures the energy (expectation value) of the quantum circuit using the cost Hamiltonian.
        
        Args:
            precision (float): Precision for the energy measurement.
        Returns:
            float: The computed energy (expectation value) of the quantum circuit with respect to the cost Hamiltonian.
        """
        
        HI = ""
        for i in range(self.n_assets): HI = HI + "I"
        H_C0 = []
        for q in range(self.n_assets): 
            Hq = HI[:q]+"Z"+HI[q+1:]
            H_C0.append((Hq, self.cost_hamiltonian_wheight(self.n_assets-1-q)))
        for e in self.edges_complete:
            He = HI[:e[0]]+"Z"+HI[e[0]+1:]
            He = He[:e[1]]+"Z"+He[e[1]+1:]
            H_C0.append((He, self.cost_hamiltonian_wheight(self.n_assets-1-e[0], 
                                                           self.n_assets-1-e[1])))
        H_C = SparsePauliOp.from_list(H_C0)
        
        estimator = StatevectorEstimator()
        job = estimator.run([(self.qc, H_C)], precision=precision)
        result = job.result()
        energy = result[0].data.evs.item()
        
        return energy
    
    def get_counts(self, shots=1000):
        """
        Measures the circuit in the computational base <shots> times and return a dict with the results
        
        Args:
            shots (int): Number of the QPU executions.
            
        Returns:
            dict: Count for each measured state.
        """
        
        qc_measured = self.qc.measure_all(inplace=False) 
        sampler = StatevectorSampler()
        job = sampler.run([qc_measured], shots=shots)
        result = job.result()
        counts = result[0].data['meas'].get_counts()
        return counts