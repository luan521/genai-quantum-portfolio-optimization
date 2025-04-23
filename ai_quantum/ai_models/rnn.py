import torch
import torch.nn as nn
import numpy as np
from ai_quantum.quantum.qaoa import QAOA

class RNN_QAOA(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, expected_value, cov_matrix, q, B, lamb, qc=None, 
                 mixture_layer='x', q_graph=None,):
        super(RNN_QAOA, self).__init__()
        self.embedding = nn.Embedding(vocab_size+1, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Quantum circuit parameters
        self.expected_value = expected_value
        self.cov_matrix = cov_matrix
        self.q = q
        self.B = B
        self.lamb = lamb
        self.qc = qc
        self.mixture_layer = mixture_layer
        self.q_graph = q_graph
        
        # Define mappings from tokens to angles for QAOA.
        self.vocab_size = vocab_size
        self.vocab_gamma = [(i/(vocab_size-1))*2*np.pi for i in range(vocab_size)]
        self.vocab_beta = [(i/(vocab_size-1))*np.pi for i in range(vocab_size)]
    
    def forward(self, x, beta_temp):
        x = self.embedding(x)
        out, hidden = self.rnn(x)
        out = out[:, -1, :]
        pred_logits = self.fc(out)
        
        pred_probabilities = torch.softmax(-beta_temp*pred_logits, dim=1)[0]
        sample_index = torch.multinomial(pred_probabilities, num_samples=1)
        
        response = pred_logits[0][sample_index], (sample_index+1).view(1, 1)
        return response
    
    def generate_parameter_sequence(self, beta_temp, depth, start_token=0):
        """
        Autoregressively generate a sequence of tokens of length 2 * depth.
        Each even-index token corresponds to a gamma and each odd-index token corresponds to a beta parameter.
        """
        x = torch.tensor([[start_token]])
        data = [self.forward(x, beta_temp)]
        for i in range(2*depth-1):
            x = torch.cat((x, data[-1][1]), dim=1)
            data.append(self.forward(x, beta_temp))
        return data
    
    def forward_qc(self, beta_temp, depth):
        data = self.generate_parameter_sequence(beta_temp, depth)
        qaoa = QAOA(self.expected_value, self.cov_matrix, self.q, self.B, self.lamb, self.qc, self.mixture_layer, self.q_graph)
        sum_w = 0
        gamma_array = []
        beta_array = []
        for i in range(depth):
            gamma = self.vocab_gamma[data[2*i][1]-1]
            beta = self.vocab_beta[data[2*i+1][1]-1]
            qaoa.add_layer(gamma, beta)
            sum_w += data[2*i][0]
            sum_w += data[2*i+1][0]
            gamma_array.append(gamma)
            beta_array.append(beta)
        
        energy, count = qaoa.measure_energy()
        
        return sum_w, torch.tensor([energy]), count, gamma_array, beta_array