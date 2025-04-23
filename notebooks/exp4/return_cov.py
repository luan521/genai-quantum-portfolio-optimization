import pandas as pd

def f_return_cov():
    """    
    Derived key metrics for assets (LIN.DE, BAYN.DE, VNE.DE, MTX.DE e MUV2.DE)
    
    Returns:
        tuple: (expected_value, cov_matrix)

    """
    expected_value = [0.26801758, -0.11724968, 0.2109537, 0.21523688, 0.1128935]
    cov_matrix = pd.DataFrame([
        [0.21117209, 0.03030933, 0.00941277, 0.02972179, 0.02922818],
        [0.03030933, 0.08796365, 0.01833403, 0.0465302,  0.04069187],
        [0.00941277, 0.01833403, 0.04971719, 0.02303918, 0.02051608],
        [0.02972179, 0.0465302,  0.02303918, 0.13717214, 0.05638483],
        [0.02922818, 0.04069187, 0.02051608, 0.05638483, 0.06765634]
    ])
    
    return expected_value, cov_matrix