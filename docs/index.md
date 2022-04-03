# antithetic
Common and Antithetic Random Variables in Python

This package provides wrappers for the ``numpy.random.Generator`` class to generate correlated sequences of random variables. Such sequences are used primarily for reducing the variance of sampling estimations. Consider approximating an expected value &mu;=E\[X\] by the average of two observations: X<sub>1</sub> and X<sub>2</sub>. In the most general setting, the variance of our estimate is 0.5(Var\[X<sub>1</sub>\] + Var\[X<sub>2</sub>\]) + Cov(X<sub>1</sub>, X<sub>2</sub>). Usually, independent samples are desirable, but enforcing Cov[X<sub>1</sub>, X<sub>2</sub>] \< 0 reduces the estimate's variance. Additional exposition can be found in Chapter 9.3 of [2].

The strategy of using correlated random variable in stochastic experiments is known as either "common random variables" or "antithetic random variables," depending on the sign of the desired correlation. For general distribution families, it can be challenging to sample from a bivariate distribution in which the marginal distributions are identical but the pair of variables is correlated. Cario and Nelson [1] describe a framework for generating the desired distribution from appropriately correlated normal random variables, for which it is simple to specify a desired correlation while maintaining a fixed marginal distribution.

A demonstration of ``antithetic``'s capabilities is available at [the project's webpage](https://njwichrowski.github.io/antithetic/elaborate_demo.html)

## Distributions Available
At present, correlated normal (``antithetic.scalars.Normal``) and continuously uniform (``antithetic.scalars.Uniform``) distributions are supported. There are plans to add further well-known scalar distributions, ideally using the strategy proposed by [1,3] that allows one to specify correlation for the resulting joint distribution, but at least via a simpler approach involving transformations of the type provided in [4]. It is also hoped to add functionality for multivariate distributions.

## Dependencies
``numpy``
``scipy``

elaborate_demo.ipynb: ``matplotlib``

## References
1. Cario, Marne C.; Nelson, Barry L. "Modeling and Generating Random Vectors with Arbitrary Marginal Distributions and Correlation Matrix," (1997). DOI: 10.1145/937332.937336
2. Kroese, D. P.; Taimre, T.; Botev, Z. I. (2011). Handbook of Monte Carlo methods. John Wiley & Sons.
3. Li, Shing Ted; Hammond, Joseph L. "Generation of Pseudorandom Numbers with Specified Univariate Distributions and Correlation Coefficients," IEEE Trans. Syst. Man Cybern.: Syst. (1975) 5, 557-561.
4. McLaughlin, Michael P. "[A Compendium of Common Probability Distributions](https://www.causascientia.org/math_stat/Dists/Compendium.pdf)," (2016).

## License
The *antithetic* package is offered under the [MIT License](http://opensource.org/licenses/MIT)

 © [Noah J. Wichrowski](https://github.com/njwichrowski), 2022-
