import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#sequence_length x batch
def randomBatch(batch_size, sequenceLength): 

    cov = np.identity(sequenceLength)

    for i in range(0, sequenceLength): 
        for j in range(0, sequenceLength): 

            if abs(i - j) < 2: 
                cov[i][j] = 0.05

            if i % 10 == j % 10: 
                cov[i][j] = 1.0


    y = []

    for i in range(0, batch_size): 
        meanDelta = np.random.randint(-5, 5)

        mean = [0.0]

        for j in range(1, sequenceLength): 
            mean += [mean[-1] + 0.01 * meanDelta]

        y_new = multivariate_normal.rvs(mean = mean, cov = cov)

        plt.plot(y_new)
        plt.show()

        y += [y_new]

    return np.asarray(y)

if __name__ == "__main__": 
    z = randomBatch(100, 200)

    #for i in range(0, 100): 
        #print z[:,i]
        #plt.plot(z[:,i])
        #plt.show()

