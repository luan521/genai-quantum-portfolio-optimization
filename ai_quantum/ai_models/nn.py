import torch
import torch.nn as nn
import numpy as np
from ai_quantum.quantum.qaoa import QAOA

class NN_QAOA(nn.Module):
    
    def __init__(self, depth, expected_value, cov_matrix, q, B, lamb, qc=None, mixture_layer='x', vocab_size=5, latent_dim=5):
        super(NN_QAOA, self).__init__()
        
        # Quantum circuit parameters
        self.expected_value = expected_value
        self.cov_matrix = cov_matrix
        self.q = q
        self.B = B
        self.lamb = lamb
        self.qc = qc
        self.mixture_layer = mixture_layer
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2*depth*vocab_size),
            nn.ReLU(),
            nn.Linear(2*depth*vocab_size, 2*depth*vocab_size)
        )
        
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
        x = self.decoder(z)
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
            for j in range(len(sampled_index_gamma)):
                gamma_array.append(self.gamma[j][sampled_index_gamma[j]])
            beta_array = []
            for j in range(len(sampled_index_beta)):
                beta_array.append(self.beta[j][sampled_index_beta[j]])
                
            qaoa = QAOA(self.expected_value, self.cov_matrix, self.q, self.B, self.lamb, self.qc, self.mixture_layer)
            for j in range(self.depth):
                qaoa.add_layer(gamma_array[j], beta_array[j])
            energy_i, count = qaoa.measure_energy()
            
            energy.append(energy_i)
        energy = torch.tensor(np.array(energy))
        
        return w, energy, count, gamma_array, beta_array