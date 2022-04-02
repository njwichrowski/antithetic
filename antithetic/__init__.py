"""
Antithetic Random Variables

This module provides a wrapper for the ``numpy.random.Generator`` class for
generating correlated sequences of random variables. Such sequences contain
pairs of values with correlation value determined by a user-specified param-
eter, though values from different pairs are independent.

For theoretical reference, see Chapter 9.3 of:
    Kroese, D. P.; Taimre, T.; Botev, Z. I. (2011). Handbook of Monte Carlo
    methods. John Wiley & Sons.
    
Dependencies
------------
numpy
scipy

"""
from numpy import array as np_array, sqrt as np_sqrt

from . import scalars
# from . import vectors # To do.

def correlation(x_data, y_data):
    """
    Compute the empirical correlation between two vectors.
    
    Parameters
    ----------
    x_data, y_data : array_like
        Vectors of paired obsevations for which the correlation is desired.
        Both arrays are flattened and must have the same size.
    """
    x = np_array(x_data).flatten()
    y = np_array(y_data).flatten()
    if x.size != y.size:
        raise ValueError("Vectors must have the same size.")
    mx = x.mean()
    my = y.mean()
    s2x = ((x - mx)**2.0).sum()
    s2y = ((y - my)**2.0).sum()
    return ((x - mx)*(y - my)).sum()/np_sqrt(s2x*s2y)
