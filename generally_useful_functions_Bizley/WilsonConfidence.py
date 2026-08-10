#Rewritten code from /r2/r2/lib/db/_sorts.pyx

### This is an implementation of Wilson Confidence interval, grabbed from a stackoverflow post which grabbed it
### from reddit code, cited above. Rather than taking in rates, it takes in counts.
### My intention now is to use it for Hit and Miss rates but it is generally applicable
### wants the z value instead of confidence percentage... 1.96 is 95%.
### the original function also didn't bother with the upper bound (because of its use case) so I put it there.

from math import sqrt

def WilsonConfidence(Ones, Zeros,z = 1.0): ### or z = 1.96...
    n = Ones + Zeros

    if n == 0:
        return 0

    phat = float(Ones) / n
    lowerBound = ((phat + z*z/(2*n) - z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))
    upperBound = ((phat + z*z/(2*n) + z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))
    return lowerBound,upperBound