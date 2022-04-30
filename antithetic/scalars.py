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

class Normal(AntitheticScalar):
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
            Mean of the marginal distribution to be sampled.
            
        scale : float
            Standard deviation of the marginal distribution to be sampled.
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

class InverseCDF(AntitheticScalar):
    """
    Common or antithetic random variables with marginal distribution specified
    via cumulative distribution function (CDF). This subclass generates
    correlated uniform random variables and applies the inverse CDF to obtain
    a desired marginal distribution.
    
    The uniform variables are derived from the raw normal random variables of
    the AntitheticScalar base class. Cario and Nelson [1] report an analytical
    relationship between the correlation of standard normal random variables
    and the uniform variables that result from transforming by inverse normal
    CDF, which we use here.
    
    Additional subclasses provide common distributions.
    """
    
    def __init__(self, correlation, func, seed = None, param_names = None):
        """
        Common or antithetic random variables generated from the standard
        uniform distribution by applying an inverse CDF.

        Parameters
        ----------
        correlation : float
            Correlation of the *underlying uniform* random values to be gener-
            ated. Must be in the range -1 <= correlation <= +1. In general,
            the resulting nonuniform distribution will have a different value
            of intra-pair correlation, and the lower bound on achievabe output
            correlations may not be -1.
        
        func : callable
            A function specifying how to transform the underlying uniform ran-
            dom variables, i.e., the inverse CDF of the desired distribution.
            
            This function should be vectorized over a single positional
            argument, corresponding to pre-transformed values, and be able to
            accept (while possibly ignoring) arbitrary keyword arguments,
            including any relevant distributional parameters.
        
        seed : {int, array_like[ints], SeedSequence}, optional
            Random seed used to initialize the pseudo-random number generator.
            Passed directly to the numpy.random.default_rng() method. If None
            (default) is provided, a seed will be automatically generated.
        
        param_names : list of str, optional
            A list of the names for distributional parameters defined at the
            subclass level. When setting an attribute with a name in this list,
            the internal flag indicating whether the next value has already
            been generated is forced to False, in order to prevent such a value
            remaining in the stream after altering the generator distribution.
        """
        self.transformation = func
        raw_correlation = 2.0*np.sin(np.pi*correlation/6.0)
        super().__init__(raw_correlation, seed, param_names)
    
    def get_next(self):
        """
        Generate the next correlated random scalar.
        """
        # Get next raw normal and convert to uniform:
        raw_normal_value = self.get_next_raw_normal()
        base_uniform_value = norm.cdf(raw_normal_value)
        
        # Apply inverse CDF and return result:
        return self.transformation(
            base_uniform_value,
            **self.distributional_parameters
        )
    
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
        # Get subsequent raw normals and convert to uniform:
        normal_values = self.get_sequence_raw_normal(N, method, mix_singles)
        base_uniform_values = norm.cdf(normal_values)
        
        # Apply inverse CDF and return results:
        return self.transformation(
            base_uniform_values,
            **self.distributional_parameters
        )

class Uniform(InverseCDF):
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
            Minimum value of the marginal distribution to be sampled.
            
        high : float
            Maximum value of the marginal distribution to be sampled. If
            high < low, swaps the values. If low == high, raises a ValueError.
        
        seed : {int, array_like[ints], SeedSequence}, optional
            Random seed used to initialize the pseudo-random number generator.
            Passed directly to the numpy.random.default_rng() method. If None
            (default) is provided, a seed will be automatically generated.
        """
        if low == high:
            raise ValueError("Degenerate distribution: low == high")
        
        self.low = float(min(low, high))
        self.high = float(max(low, high))
        
        super().__init__(
            correlation = correlation,
            func = Uniform.inverse_CDF,
            seed = seed,
            param_names = ["low", "high"]
        )
    
    def change_correlation(self, new_correlation):
        """
        Specify a new value for the intra-pair correlation of the marginally
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
        
        # Update raw correlation. The setter method defined in base class
        # AntitheticScalar handles resetting the generation flags:
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
    def correlation(self):
        return 6.0*np.arcsin(0.5*self.raw_correlation)/np.pi
    
    @classmethod
    def inverse_CDF(cls, u, low = 0.0, high = 1.0, **kwargs):
        return low + (high - low)*u

class Exponential(InverseCDF):
    """
    Common or antithetic exponential random variables.
    """
    
    def __init__(self, correlation, loc = 0.0, scale = 1.0, rate = None,
                 seed = None, corr_for_unif = True):
        """
        Common or antithetic exponential random variables with specified
        marginal distribution. See base.AntitheticScalar for more details.
        
        Parameters
        ----------
        correlation : float
            Correlation of common or antithetic values to be generated. Must
            be in the range -1 <= correlation <= +1. Depending on the value of
            corr_for_unif, this parameter may specify the correlation either
            of the underlying uniform random varibles or of the exponential
            random variables themselves.
        
        loc : float
            Minimum value of the marginal distribution's support. Exponential
            distributions are almost always restricted to loc = 0.0, but this
            parameter allows for a horizontal shift.
        
        scale : float
            If loc == 0.0, the mean value for the marginal distribution to be
            sampled. In general, the expected value is loc + scale.
            
        rate : float, optional
            If provided, assigns scale = 1.0/rate, superceding the value
            received for scale. This allows one to specify a distribution
            under the rate parameter framework.
        
        seed : {int, array_like[ints], SeedSequence}, optional
            Random seed used to initialize the pseudo-random number generator.
            Passed directly to the numpy.random.default_rng() method. If None
            (default) is provided, a seed will be automatically generated.
        
        corr_for_unif : bool, optional
            If True (default), the value of correlation refers to the uniform
            distribution from which the exponential variates are derived. If
            False, determines a value for the uniform correlation that yields
            exponential variates with (approimately) the desired correlation.
                > The latter case is not yet supported.
        """
        if not corr_for_unif:
            raise NotImplementedError(
                "Direct values for exponential correlation not yet supported."
            )
        
        if rate is not None:
            scale = 1.0/rate
        if scale <= 0.0:
            raise ValueError("Invalid scale parameter: %r" % (scale,))
        self.scale = float(scale)
        self.loc = float(loc)
        
        if corr_for_unif:
            super().__init__(
                correlation = correlation,
                func = Exponential.inverse_CDF,
                seed = seed,
                param_names = ["loc", "scale"]
            )
    
    @property
    def mean(self):
        return self.loc + self.scale
    
    @property
    def standard_deviation(self):
        return self.scale
    
    @property
    def variance(self):
        return self.scale**2.0
    
    @property
    def rate(self):
        return 1.0/self.scale
    
    @classmethod
    def inverse_CDF(cls, u, loc = 0.0, scale = 1.0, **kwargs):
        return loc - scale*np.log(1.0 - u)