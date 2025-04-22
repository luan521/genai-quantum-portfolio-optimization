# Test 1
- **Objective**: Validate the proposal of a deep‑learning model trained to generate QAOA parameters for portfolio optimization, assessing its feasibility and—if successful—examining how quantum‑circuit depth affects the solution. The model is a neural network with three linear layers separated by ReLU activation functions (code link). As a benchmark, a standard QAOA parameter optimization was run with the classical COBYLA algorithm (code link).
- **Conclusion**: With COBYLA, convergence was only reached when the circuit depth was at least 8. In contrast, the neural‑network method converged with a depth of 1, demonstrating a clear efficiency gain. However, for depths greater than 3 the neural‑network approach no longer converged.

# Test 2
- **Objective**: To address the convergence issue for circuit depths above 3 observed in Test 1, a recurrent neural network implementation was evaluated (code link).
- **Conclusion**: The test was successful. A key advantage of the recurrent approach is its sequential nature: training for a larger depth can start from a pre‑training phase at a lower depth. This strategy resolved the convergence problem seen at higher depths in Test 1.