"""
Subclasses of antithetic.base.AntitheticScalar for univariate distributions.

References
----------
[1] Cario, Marne C.; Nelson, Barry L. ``Modeling and Generating Random Vectors
    with Arbitrary Marginal Distributions and Correlation Matrix,'' (1997).
    DOI: 10.1145/937332.937336

"""

import numpy as np
from scipy.stats import norm

from .base import AntitheticScalar

class AntitheticNormal(AntitheticScalar):
    """
    Common or antithetic Gaussian random variables.
    """
    
    def __init__(self, correlation, loc = 0.0, scale = 1.0, seed = None):
        """
        Common or antithetic Gaussian random variables with specified marginal
        mean and variance. See base.AntitheticScalar for more details.
        
        Parameters
        ----------
        correlation : float
            Correlation of common or antithetic values to be generated. Must
            be in the range -1 <= correlation <= +1.
        
        loc : float
            Mean of the marginal distribution to be generated.
            
        scale : float
            Standard deviation of the marginal distribution to be generated.
            Must be a positive value.
        
        seed : {int, array_like[ints], SeedSequence}, optional
            Random seed used to initialize the pseudo-random number generator.
            Passed directly to the numpy.random.default_rng() method. If None
            (default) is provided, a seed will be automatically generated.
        """
        if scale <= 0.0:
            raise ValueError("Invalid scale parameter: %r" % (scale,))
        
        self.loc = float(loc)
        self.scale = float(scale)
        super().__init__(correlation, seed, param_names = ["loc", "scale"])
    
    @property
    def mean(self):
        return self.loc
    
    @property
    def standard_deviation(self):
        return self.scale
    
    @property
    def variance(self):
        return self.scale**2.0
    
    @property
    def correlation(self):
        return self.raw_correlation
    
    def get_next(self):
        """
        Generate the next correlated Gaussian random scalar.
        """
        return self.scale*self.get_next_raw_normal() + self.loc
    
    def get_sequence(self, N, method = "zip", mix_singles = True):
        """
        Generate multiple subsequent values in a sequence of correlated
        Gaussian random scalars, as if calling get_next_raw_normal the same
        number of times and collecting the results. Optionally rearrange the
        values before returning, so that paired values are not adjacent.
        
        Parameters
        ----------
        N : int
            The number of values to generate.
        method : {"zip", "shuffle", "concatenate"}, optional
            Procedure for assembling the overall sequence from two paired
            subsequences with correlated entries.  (Default: "zip")
                "zip" : Combine subsequences entry-by-entry, so that each item
                        is adjacent to the other member of its pair.
                "shuffle" : Randomly permute all items.
                "concatenate" : Combine the subsequences end-to-end.
        mix_singles : bool, optional
            If True (default) and method == "shuffle", include unpaired values
            on the beginning or end of the sequence, if present, in the
            shuffling procedure. If False, such values will be kept in-place
            while shuffling the interior pairs entirely contained in sequence.
        
        Returns
        -------
        seq : np.ndarray (N,)
            The sequence of correlated, marginally normal scalars.
        """
        sequence = self.get_sequence_raw_normal(N, method, mix_singles)
        return self.scale*sequence + self.loc

class AntitheticUniform(AntitheticScalar):
    """
    Common or antithetic uniform random variables.
    """
    
    def __init__(self, correlation, low = 0.0, high = 1.0, seed = None):
        """
        Common or antithetic uniform random variables with specified marginal
        support. See base.AntitheticScalar for more details.
        
        Parameters
        ----------
        correlation : float
            Correlation of common or antithetic values to be generated. Must
            be in the range -1 <= correlation <= +1. Cario and Nelson [1]
            report an analytical relationship between the correlation of
            standard normal random variables and the uniform variables that
            result from transforming by inverse normal CDF, which we use here.
        
        low : float
            Minimum value of the marginal distribution to be generated.
            
        high : float
            Maximum value of the marginal distribution to be generated. If
            high < low, swaps the values. If low == high, raises a ValueError.
        
        seed : {int, array_like[ints], SeedSequence}, optional
            Random seed used to initialize the pseudo-random number generator.
            Passed directly to the numpy.random.default_rng() method. If None
            (default) is provided, a seed will be automatically generated.
        """
        if low == high:
            raise ValueError("Degenerate distribution: low == high")
        
        self.low = min(low, high)
        self.high = max(low, high)
        
        raw_correlation = 2.0*np.sin(np.pi*correlation/6.0)
        super().__init__(raw_correlation, seed, param_names = ["low", "high"])
    
    def get_next(self):
        """
        Generate the next correlated uniform random scalar.
        """
        raw_normal_value = self.get_next_raw_normal()
        base_uniform_value = norm.cdf(raw_normal_value)
        return self.range*base_uniform_value + self.low
    
    def get_sequence(self, N, method = "zip", mix_singles = True):
        """
        Generate multiple subsequent values in a sequence of correlated
        uniform random scalars, as if calling get_next_raw_normal the same
        number of times and collecting the results. Optionally rearrange the
        values before returning, so that paired values are not adjacent.
        
        Parameters
        ----------
        N : int
            The number of values to generate.
        method : {"zip", "shuffle", "concatenate"}, optional
            Procedure for assembling the overall sequence from two paired
            subsequences with correlated entries.  (Default: "zip")
                "zip" : Combine subsequences entry-by-entry, so that each item
                        is adjacent to the other member of its pair.
                "shuffle" : Randomly permute all items.
                "concatenate" : Combine the subsequences end-to-end.
        mix_singles : bool, optional
            If True (default) and method == "shuffle", include unpaired values
            on the beginning or end of the sequence, if present, in the
            shuffling procedure. If False, such values will be kept in-place
            while shuffling the interior pairs entirely contained in sequence.
        
        Returns
        -------
        seq : np.ndarray (N,)
            The sequence of correlated, marginally uniform scalars.
        """
        normal_values = self.get_sequence_raw_normal(N, method, mix_singles)
        base_uniform_values = norm.cdf(normal_values)
        return self.range*base_uniform_values + self.low
    
    def change_correlation(self, new_correlation):
        """
        Specify a new value for the within-pair correlation of the marginally
        uniform variables to generate. Computes the required raw_correlation
        for the underlying normal variables and updates accordingly.
        
        Parameters
        ----------
        correlation : float
            New correlation of common or antithetic values to be generated.
            Must be in the range -1 <= new_correlation <= +1.
        """
        if not(-1.0 <= new_correlation <= 1.0):
            raise ValueError("Invalid correlation: %r" % (new_correlation,))
        
        self.raw_correlation = 2.0*np.sin(np.pi*new_correlation/6.0)
    
    @property
    def mean(self):
        return 0.5*(self.low + self.high)
    
    @property
    def standard_deviation(self):
        return 0.28867513459481287*(self.high - self.low)
    
    @property
    def variance(self):
        return (self.high - self.low)**2.0/12.0
    
    @property
    def range(self):
        return self.high - self.low
    
    @property
    def correlation(self):
        return 6.0*np.arcsin(0.5*self.raw_correlation)/np.pi
