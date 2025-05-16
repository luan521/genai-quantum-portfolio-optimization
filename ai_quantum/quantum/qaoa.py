from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

class QAOA():
    """
    Implements QAOA for portfolio optimization using a quantum circuit to solve QUBO problems.
    
    Attributes:
        expected_value (list[float]): Expected returns for each asset.
        cov_matrix (DataFrame-like): Covariance matrix between assets.
        q (float): Scaling factor for covariance.
        B (float): Budget/threshold parameter.
        lamb (float): Penalty factor.
        n_assets (int): Number of assets/qubits.
        qc (qiskit.circuit.quantumcircuit.QuantumCircuit): Quantum circuit with initial initialization
    """
    
    def __init__(self, expected_value, cov_matrix, q, B, lamb, qc=None, mixture_layer='x', q_graph=None):
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
            q_graph (list): list of tuples containing all connected qubits
        """
        
        self.q = q
        self.B = B
        self.lamb = lamb
        self.expected_value = expected_value
        self.cov_matrix = cov_matrix
        self.n_assets = len(expected_value)
        self.mixture_layer = mixture_layer
        
        if qc is not None:
            self.qc = qc.copy()
        else:
            self.qc = QuantumCircuit(self.n_assets, self.n_assets)
            # Initialization - prepare an equal superposition state
            for qubit in range(self.n_assets):
                self.qc.h(qubit)
            self.qc.barrier()
            
        if q_graph is None:
            self.q_graph = [(i, j) for i in range(5) for j in range(i+1, 5)]
        else:
            self.q_graph = q_graph
    
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
            response =2*self.expected_value[i]+2*self.lamb*(2*self.B-self.n_assets)-self.q*self.cov_matrix[i].sum()
        else:
            response = self.q*self.cov_matrix[i][j]+2*self.lamb
        return response
    
    def draw(self):
        """
        Draws the quantum circuit using matplotlib.
        """
        self.qc.draw(output="mpl", style="iqp")
        
    def _add_mixture_layer(self, beta):
        """
        Implement exp(-i*beta*H_B).
        """
        
        if self.mixture_layer == 'x':
            for qubit in range(self.n_assets):
                self.qc.rx(2*beta, qubit)
        elif self.mixture_layer == 'ring_mixer':
            for e in self.q_graph:
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
        for e in self.q_graph:
            self.qc.cx(e[0], e[1])
            self.qc.rz(2*gamma*self.cost_hamiltonian_wheight(e[0], e[1]), e[1])
            self.qc.cx(e[0], e[1])
        for qubit in range(self.n_assets):
            self.qc.rz(2*gamma*self.cost_hamiltonian_wheight(qubit), qubit)
        self.qc.barrier()
                
        # Implement exp(-i*beta*H_B)
        self._add_mixture_layer(beta)
        self.qc.barrier()
             
    def measure_energy(self):
        """
        Measures the circuit and computes the expected energy.
        
        Returns:
            tuple: (energy, counts) where energy is the expected value and counts is the measurement distribution.
    
        """
        
        self.qc.measure(range(self.n_assets), range(self.n_assets))

        simulator = AerSimulator()
        compiled_circuit = transpile(self.qc, simulator)
        sim_result = simulator.run(self.qc).result()
        counts = sim_result.get_counts()
        
        energy = 0
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            prob = count / total_shots
            
            Z = [1 if bitstring[::-1][i] == '0' else -1 for i in range(self.n_assets)]
            
            energy_outcome = 0
            for i in range(self.n_assets):
                energy_outcome += self.cost_hamiltonian_wheight(i)*Z[i]
                for j in range(i+1, self.n_assets):
                    energy_outcome += self.cost_hamiltonian_wheight(i, j)*Z[i]*Z[j]
            
            energy += prob * energy_outcome
    
        return energy, counts