import torch
import torch.nn as nn
import numpy as np
from ai_quantum.quantum.qaoa import QAOA

class CNN_QAOA(nn.Module):

    def __init__(self, depth, expected_value, cov_matrix, q, B, lamb, qc=None, mixture_layer='x', q_graph=None,
                 vocab_size=5, latent_dim=5, kernel_size=3):
        super(CNN_QAOA, self).__init__()
        
        # Quantum circuit parameters
        self.expected_value = expected_value
        self.cov_matrix = cov_matrix
        self.q = q
        self.B = B
        self.lamb = lamb
        self.qc = qc
        self.mixture_layer = mixture_layer
        self.q_graph = q_graph
        
        if kernel_size % 2 != 1:
            kernel_size += 1
        padding = int((kernel_size-1)/2)
        
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 2*depth*vocab_size)
        self.fc3 = nn.Linear(2*depth*vocab_size, 2*depth*vocab_size)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding)
        
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.depth = depth
        
        # Define mappings from tokens to angles for QAOA.
        self.gamma = [
                      [(i/(vocab_size-1))*2*np.pi for i in range(vocab_size)]
                      for _ in range(depth)
                     ]
        self.beta = [
                     [(i/(vocab_size-1))*np.pi for i in range(vocab_size)]
                     for _ in range(depth)
                    ]

    def forward(self, z, beta_temp=5):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = x.view(-1, 1, self.vocab_size)
        x = self.conv1(x)
        x = x.view(-1, 2*self.depth*self.vocab_size)
        x = self.fc3(x)
        W = x.view(-1, self.vocab_size)
        
        probabilities = torch.softmax(-beta_temp*W, dim=-1)
        sample_index = torch.multinomial(probabilities, num_samples=1)
        w = W[torch.arange(W.size(0)), sample_index.view(sample_index.size(0))].view(x.size(0), -1)
        sampled_index = sample_index.view(x.size(0), -1)
        
        energy = []
        for i in range(sampled_index.size(0)):
            sampled_index_i = sampled_index[i]
            gamma_array = []
            sampled_index_gamma = sampled_index_i[:self.depth]
            sampled_index_beta = sampled_index_i[self.depth:]
            for i in range(len(sampled_index_gamma)):
                gamma_array.append(self.gamma[i][sampled_index_gamma[i]])
            beta_array = []
            for i in range(len(sampled_index_beta)):
                beta_array.append(self.beta[i][sampled_index_beta[i]])
                
            qaoa = QAOA(self.expected_value, self.cov_matrix, self.q, self.B, self.lamb, self.qc, self.mixture_layer, self.q_graph)
            for i in range(self.depth):
                qaoa.add_layer(gamma_array[i], beta_array[i])
            energy_i, count = qaoa.measure_energy()
            
            energy.append(energy_i)
        energy = torch.tensor(np.array(energy))
        
        return w, energy, count, gamma_array, beta_array