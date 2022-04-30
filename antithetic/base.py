"""
Base classes for generating pair-correlated random variables.
"""

import numpy as np

def bivariate_covariance_matrix(rho, square_root = False):
    """
    Generate a 2-by-2 covariance matrix (or its Cholesky square root) for
    a pair of variables with unit variance and specified covariance.
    
    Parameters
    ----------
    rho : float
        Correlation between random variables represented by the covariance
        matrix. Must be in the range -1 <= rho <= +1.
    
    square_root : bool, optional
        If True (default: False), return the lower-diagonal matrix square
        root, A, such that numpy.matmul(A, A.T) is the covariance matrix.
    """
    if not(-1.0 <= rho <= 1.0):
        raise ValueError("Invalid correlation: %r" % (rho,))
    
    if square_root:
        return np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho**2.0)]])
    else:
        return np.array([[1.0, rho], [rho, 1.0]])

class AntitheticScalar(object):
    """ Superclass for antithetic random scalars. """
    
    def __init__(self, raw_correlation, seed = None, param_names = None):
        """
        Container for numpy.random.Generator objects that provides support
        for generating antithetic sequences of random variables. This is a
        base class that generates correlated, jointly normal Gaussian random
        variables having standardized marginals (zero mean, unit variance).
        Subclasses provide distributional transformations.
        
        Parameters
        ----------
        raw_correlation : float
            Value to be used for the correlation of common (positive) or anti-
            thetic (negative) pairs in the underlying standard normal distri-
            bution from which other distributions, defined as subclasses, are
            utlimately derived. Must satisfy -1 <= raw_correlation <= +1.
        
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
        if not(-1.0 <= raw_correlation <= 1.0):
            raise ValueError("Invalid correlation: %r" % (raw_correlation,))
        self.raw_correlation = float(raw_correlation)
        self.generator = np.random.default_rng(seed)
        
        if param_names is None:
            self.distributional_parameter_names = list()
        else:
            self.distributional_parameter_names = param_names
        if "raw_correlation" not in self.distributional_parameter_names:
            self.distributional_parameter_names.append("raw_correlation")
        
        self.__have = False # Track whether a new value must be generated.
        self.__hold = None # Storage for generated, to-be-used value(s).
    
    def __repr__(self):
        """ Generic ``repr'' function. Overwritable at subclass level. """
        return "%s(raw_correlation = %f)" % (
            self.__class__.__name__,
            self.raw_correlation
        )
    
    def __str__(self):
        """ Generic ``str'' function. Overwritable at subclass level. """
        return "%s(raw_correlation = %f)" % (
            self.__class__.__name__,
            self.raw_correlation
        )
    
    def __setattr__(self, name, value):
        """ Set attribute value, with a few operational checks. """
        
        # Prevent an impossible value being assigned to raw_correlation:
        if name == "raw_correlation" and not(-1.0 <= value <= 1.0):
            raise ValueError("Invalid correlation: %r" % (value,))
        
        # If changing a distributional parameter, prevent a stored value in
        # self.__hold from being used on the next call, since this would be
        # from the old distribution. (Check before the usual setting because
        # it's less restrictive on subclasses' __init__ methods):
        try:
            if name in self.distributional_parameter_names:
                super().__setattr__("__have", False)
        except AttributeError:
            super().__setattr__("distributional_parameter_names", list())
        
        # Set attribute value as normal:
        super().__setattr__(name, value)
    
    @property
    def raw_covariance_matrix(self):
        return bivariate_covariance_matrix(self.raw_correlation)
    
    @property
    def mixing_weights(self):
        """
        Antithetic weight vector: if X, Y are iid standard normal and we
        define Z = rho*X + sqrt(1 - rho**2.0)*Y, then Z is standard normal
        with Cov(X, Z) = Corr(X, Z) = rho.
        """
        return np.array([
            self.raw_correlation,
            np.sqrt(1.0 - self.raw_correlation**2.0)
        ])
    
    @property
    def distributional_parameters(self):
        """ Construct a dictionary of distribution parameter values. """
        d = dict()
        for key in self.distributional_parameter_names:
            d[key] = getattr(self, key, None)
        return d
    
    def set_seed(self, seed = None):
        """
        Create a new numpy.random.Generator object, using the specified seed,
        to use for pseudo-random number generation.
        
        Parameters
        ----------
        seed : {int, array_like[ints], SeedSequence}, optional
            Random seed used to initialize the pseudo-random number generator.
            Passed directly to the numpy.random.default_rng() method. If None
            (default) is provided, a seed will be automatically generated.
        """
        self.generator = np.random.default_rng(seed)
        self.__have = False # Don't use a stored value after resetting seed.

    def get_next_raw_normal(self):
        """
        Generate the next value in a sequence of correlated Gaussian
        random scalars. On each call to the method, returns a float that has
        a standard normal distribution. However, values returned by sequential
        pairs of calls have correlation self.raw_correlation. Independence
        is preserved for calls not paired together, such that running:
        
            g = AntitheticScalar(rho, seed = 1)
            vector = list()
            for i in range(4):
                vector.append(g.get_next_raw_normal())
        
        produces a four-entry vector with covariance matrix
        
                    [[1.0 rho 0.0 0.0]
                     [rho 1.0 0.0 0.0]
                     [0.0 0.0 1.0 rho]
                     [0.0 0.0 rho 1.0]].
        """
        # Use a previously stored value (second member of pair) if available:
        if self.__have:
            self.__have = False
            return self.__hold
        
        # Otherwise, generate a new pair, return one, and store the other:
        result = self.generator.normal(size = 2)
        next_value = result[0]
        self.__hold = np.sum(result*self.mixing_weights)
        
        self.__have = True
        return next_value
    
    def get_sequence_raw_normal(self, N, method = "zip", mix_singles = True):
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
            subsequences with correlated entries. (Default: "zip")
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
            The sequence of correlated, marginally standard normal scalars.
        """
        ### To do: use a second generator for shuffling randomness? ###
        
        if method not in ["zip", "shuffle", "concatenate"]:
            raise ValueError("Unrecognized method: %r" % (method,))
        
        if N <= 0:
            raise ValueError("Cannot generate %d values." % (N,))
        elif N == 1:
            return self.get_next_raw_normal()
        values = np.zeros(N)
        
        # Start with stored value, if available:
        front_single = self.__have
        if front_single:
            values[0] = self.__hold
        
        # Determine whether to store an unpaired value:
        back_single = bool((N - int(front_single)) % 2)
        
        # Generate interior pairs:
        M = N - int(front_single) - int(back_single)
        assert M % 2 == 0 # To do: Remove? Switch to if-raise?
        
        pairs = self.generator.normal(size = (M//2, 2))
        pairs[:,1] = np.sum(pairs*self.mixing_weights, axis = 1)
        
        # If needed, generate another pair and store extra value:
        self.__have = back_single
        if back_single:
            result = self.generator.normal(size = 2)
            values[-1] = result[0]
            self.__hold = np.sum(result*self.mixing_weights)
        
        # Arrange generated values in requested method:
        if method == "zip":
            values[int(front_single):N-int(back_single)] = pairs.flatten()
        else: # Concatenate sequences, then shuffle if requested:
            values[int(front_single):N-int(back_single)] = pairs.T.flatten()
            
            if method == "shuffle":
                if mix_singles: # Shuffle all values:
                    idx = self.generator.permutation(N)
                    values = values[idx]
                else: # Shuffle only the internal pairs:
                    idx = int(front_single) + self.generator.permutation(M)
                    values[int(front_single):N-int(back_single)] = values[idx]
        
        return values
