import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

def cdflst(sortedSamples): 

    cumsum = sum(sortedSamples)
    cdf = []

    psum = 0.0

    for i in range(0, len(sortedSamples)): 
        psum += sortedSamples[i]
        cdf += [erfinv(2.0 * i / len(sortedSamples) - 1)]
        

    return cdf

#Given two lists of samples (not necessarily sorted), plots the CDF of each.  
def qqplot(samples1, samples2): 
    samples1 = sorted(samples1)
    samples2 = sorted(samples2)

    cdf1 = cdflst(samples1)
    cdf2 = cdflst(samples2)

    plt.scatter(samples1, cdf1, color = 'red')
    plt.scatter(samples2, cdf2, color = 'blue')


    plt.title("samples red, true dist blue")

    plt.show(block=True)

if __name__ == "__main__": 
    
    x1 = np.random.normal(3, 0.4, 10000)
    x2 = np.random.normal(3, 2, 10000)

    qqplot(x1, x2)
