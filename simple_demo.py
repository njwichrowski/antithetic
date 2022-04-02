from scipy.stats import norm

import antithetic

num_pairs = 10000
corr = -0.5

rngN = antithetic.scalars.Normal(
    correlation = corr,
    loc = 0.0,
    scale = 1.0,
    seed = 1
)

rngU = antithetic.scalars.Uniform(
    correlation = corr,
    low = 0.0,
    high = 1.0,
    seed = 2
)

####################################
### Demonstrate AntitheticNormal ###
####################################
print("Generating %d pairs of standard normal\n"
      "variables with in-pair correlation %.5f.\n" % (num_pairs, corr))
seqN = rngN.get_sequence(2*num_pairs)

print("Sample Mean:     %8.5f" % (seqN.mean(),))
print("Sample Variance: %8.5f" % (seqN.std(ddof = 1)**2.0,))

rho_within = antithetic.correlation(seqN[::2], seqN[1::2])
rho_across = antithetic.correlation(seqN[1:-1:2], seqN[2::2])
print("Sample Correlation, Within Pairs: %8.5f" % (rho_within,))
print("Sample Correlation, Across Pairs: %8.5f" % (rho_across,))

print("\nPartial Histogram Data\n  Bin   | Exp. | Obs.")
print("--------+------+-----")
for i in range(-3, 3):
    line = "%+2d<X<%+2d | " % (i, i+1)
    
    prob = norm.cdf(i+1) - norm.cdf(i)
    line += "%4d | " % (prob*seqN.size,)
    
    count = (seqN <= i+1).sum() - (seqN <= i).sum()
    line += "%4d" % count
    
    print(line)

#####################################
### Demonstrate AntitheticUniform ###
#####################################
print("\n\n\nGenerating %d pairs of standard uniform\n"
      "variables with in-pair correlation %.5f.\n" % (num_pairs, corr))
seqU = rngU.get_sequence(2*num_pairs)

print("Sample Mean:     %8.5f" % (seqU.mean(),))
print("Sample Variance: %8.5f" % (seqU.std(ddof = 1)**2.0,))

rho_within = antithetic.correlation(seqU[::2], seqU[1::2])
rho_across = antithetic.correlation(seqU[1:-1:2], seqU[2::2])
print("Sample Correlation, Within Pairs: %8.5f" % (rho_within,))
print("Sample Correlation, Across Pairs: %8.5f" % (rho_across,))

seqU = seqU.reshape((num_pairs, 2)).T
small_small = ((seqU[0] < 0.5)*(seqU[1] < 0.5)).sum()
large_small = ((seqU[0] > 0.5)*(seqU[1] < 0.5)).sum()
small_large = ((seqU[0] < 0.5)*(seqU[1] > 0.5)).sum()
large_large = ((seqU[0] > 0.5)*(seqU[1] > 0.5)).sum()
print("\nQuadrant Counts Within Pairs")
print("X1 < 0.5, X2 < 0.5 | %4d" % (small_small,))
print("X1 > 0.5, X2 < 0.5 | %4d" % (large_small,))
print("X1 < 0.5, X2 > 0.5 | %4d" % (small_large,))
print("X1 > 0.5, X2 > 0.5 | %4d" % (large_large,))