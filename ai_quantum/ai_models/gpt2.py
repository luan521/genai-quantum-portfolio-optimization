import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
import numpy as np
from ai_quantum.quantum.qaoa import QAOA

class GPT2_QAOA(nn.Module):
    def __init__(self, vocab_size, max_depth, expected_value, cov_matrix, q, B, lamb, qc=None, mixture_layer='x',
                 n_embd=256, n_layer=6, n_head=8):
        super(GPT2_QAOA, self).__init__()
        config = GPT2Config(
            vocab_size=vocab_size + 1,   
            n_positions=max_depth * 2,     
            n_ctx=max_depth * 2,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Quantum circuit parameters
        self.expected_value = expected_value
        self.cov_matrix = cov_matrix
        self.q = q
        self.B = B
        self.lamb = lamb
        self.qc = qc
        self.mixture_layer = mixture_layer

        # Define mappings from tokens to angles for QAOA.
        self.vocab = [(i / (vocab_size - 1)) * 2 * np.pi for i in range(vocab_size)]
    
    def forward(self, input_ids, beta_temp):
        outputs = self.transformer(input_ids)
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states[:, -1, :]
        logits = self.lm_head(hidden_states)
        logits = logits.view(1, -1)
        
        pred_probabilities = torch.softmax(-beta_temp*logits, dim=1)[0]
        sample_index = torch.multinomial(pred_probabilities, num_samples=1)
        
        response = logits[0][sample_index], (sample_index+1).view(1, 1)
        return response
    
    def generate_parameter_sequence(self, beta_temp, depth, start_token=0):
        """
        Autoregressively generate a sequence of tokens of length 2 * depth.
        Each even-index token corresponds to a gamma and each odd-index token corresponds to a beta parameter.
        """
        input_ids = torch.tensor([[start_token]], dtype=torch.long)
        data = [self.forward(input_ids, beta_temp)]
        for i in range(2*depth-1):
            input_ids = torch.cat((input_ids, data[-1][1]), dim=1)
            data.append(self.forward(input_ids, beta_temp))
        return data

    def forward_qc(self, beta_temp, depth):
        data = self.generate_parameter_sequence(beta_temp, depth)
        qaoa = QAOA(self.expected_value, self.cov_matrix, self.q, self.B, self.lamb, self.qc, self.mixture_layer)
        sum_w = 0
        gamma_array = []
        beta_array = []
        
        for i in range(depth):
            gamma = self.vocab[data[2*i][1]-1]
            beta = self.vocab[data[2*i+1][1]-1]
            qaoa.add_layer(gamma, beta)
            sum_w += data[2*i][0]
            sum_w += data[2*i+1][0]
            gamma_array.append(gamma)
            beta_array.append(beta)
        
        energy, count = qaoa.measure_energy()
        
        return sum_w, torch.tensor([energy]), count, gamma_array, beta_array