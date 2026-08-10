import scipy

def calculate_dprime(HR,FAR,TT,RT):
    ###
    # Takes as input a hit rate, false alarm rate, target total and reference total
    # and outputs dprime. Function is intended to take into account cases where hit
    # or FA is 1 or 0, as those cases are annoying to rewrite all the time.
    # I think it can also take in CR rate and miss rate, but you need to treat CR as hit
    # and Miss as FA.
    if (HR == 1) & (FAR == 1):
        dprime = 0
    elif (HR == 0) & (FAR == 0):
        dprime = 0
    else:
        if HR == 1:
            HR = (TT-1)/TT

        if FAR == 1:
            FAR = (RT-1)/RT

        if HR == 0:
            HR = 1/TT

        if FAR == 0:
            FAR = 1/RT

        dprime = scipy.stats.norm.ppf(HR) - scipy.stats.norm.ppf(FAR)
    return dprime