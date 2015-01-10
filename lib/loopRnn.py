import theano
import theano.tensor as T
import numpy
import sys
import time
import LSTMLayer

print "run"

#Doing demo to study performance difference between for loop and LSTM on cpu.  Will also try gpu.  

#Provide two methods.  One gets a network that takes an X sequence and outputs p(X).  
def discriminator(X): 


    return params, p_X

#Takes a matrix of noise variables, loads these into memory initially (concatenated with blank memory spots).  
#Outputs an X sequence.  
def generator(): 

    return params, X


sys.setrecursionlimit(100000)


if __name__ == "__main__": 

    print "RUN"

    weightSize = 500

    lstm = LSTMLayer.LSTMLayer(weightSize, weightSize, weightSize, weightSize, 0.1, useReluReadGate = True)

    W = theano.shared(numpy.random.normal(size = (weightSize,weightSize)))

    print W.shape

    def oneStep(prevH, prevMemory, prevController): 
        #return T.dot(W, prevH) + T.dot(W, prevMemory) + T.dot(W, prevController), T.dot(W, prevMemory), T.dot(W, prevController)

        controller1, memory1, h1 = lstm.getOutputs(prevController, prevMemory, prevH)

        return controller1, memory1, h1

    sequenceLength = 200

    print "Sequence Length", sequenceLength, "Number of Hidden Units:", weightSize

    h0 = T.matrix()
    memory_0 = T.matrix()
    controller_0 = T.matrix()
    h1 = h0

    batch_size = 400

    new_h = [h0]
    new_memory = [memory_0]
    new_controller = [controller_0]

    new_h_scan, _ = theano.scan(oneStep, sequences = [], outputs_info = [h1, memory_0, controller_0], n_steps = sequenceLength)

    #print "starting grad for loop"
    #timeStart = time.time()
    #g = T.grad(sum(map(T.sum, new_h)), lstm.params)
    #print "time spent on for loop grad", time.time() - timeStart

    timeStart = time.time()
    g_scan = T.grad(T.sum(new_h_scan), lstm.params)
    print "time spent on scan grad", time.time() - timeStart

    #timeStart = time.time()
    #f = theano.function(inputs = [h0, memory_0], outputs = [new_h[-1]] + g)
    #print "time spent compling for loop", time.time() - timeStart

    timeStart = time.time()
    f_scan = theano.function(inputs = [h0, memory_0, controller_0], outputs = [new_h_scan[-1]] + g_scan)
    print "time spent compiling scan", time.time() - timeStart

    numIter = 20

    #timeStart = time.time()
    #for i in range(0, numIter): 
    #    f([[1.0] * weightSize] * batch_size, [[1.0] * weightSize] * batch_size)
    #print "time", time.time() - timeStart

    timeStart = time.time()

    for i in range(0, numIter): 
        f_scan([[1.0] * weightSize] * batch_size, [[1.0] * weightSize] * batch_size, [[1.0] * weightSize] * batch_size)

    print "time for scan version", time.time() - timeStart














