import pandas as pd

def f_return_cov():
    """    
    Derived key metrics for assets (BRKM5.SA, ITUB4.SA, VALE3.SA, KLBN4.SA) over the period from 2016-01-01 to 2021-09-20
    
    Returns:
        tuple: (expected_value, cov_matrix)

    """
    expected_value = [0.335649, 0.084554, 0.357477, 0.148336]
    cov_matrix = 252*pd.DataFrame([
     [0.001077, 0.000257,0.000320,0.000190],
     [0.000257,	0.000441,0.000228,0.000084],
     [0.000320,	0.000228, 0.000867,0.000154],
     [0.000190,	0.000084,0.000154,0.000420]
    ])
    
    return expected_value, cov_matrix