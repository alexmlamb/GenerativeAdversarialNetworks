import sys
import matplotlib.pyplot as plt
import linecache
import random
import utils
import numpy as np


def getData(): 
    specialStr = str(hash("a"))

    yStack = []

    tsLst = []

    for i in range(1, 10000): 
        line = linecache.getline(sys.argv[1], random.randint(1,2000))
        line = utils.uncompress(line)

        y = np.asarray(line["TimeSeriesTarget"], dtype = 'float32')

        yStack += [y]

        if len(yStack) == 100: 
            yLst = np.vstack(yStack).swapaxes(0,1)
            yLst = yLst.reshape(yLst.shape[0], yLst.shape[1], 1)
            #print yLst.shape
            yStack = []
            tsLst += [yLst]

    return tsLst






