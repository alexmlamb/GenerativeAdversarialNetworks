import theano
import theano.tensor as T
import numpy
import sys
import time
import LSTMLayer
import numpy as np

print "run"

'''

Build discriminator, generator.  Create sample sequences from multivariate normal distribution with linear trend.  Also consider using fresh demand?  First train to generate sequence without conditioning on part of sequence?  

'''

#Provide two methods.  One gets a network that takes an X sequence and outputs p(X).  
def discriminator(X): 

    lstm = LSTMLayer.LSTMLayer(inputSize = 1, controllerSize = 500, memorySize = 500, outputSize = 500, initialScale = 0.05)

    h1 = T.matrix()
    memory_0 = T.matrix()
    output_0 = T.matrix()

    def oneStep(prevMemory, prevOutput, inputSequence):
        memory, output = lstm.getOutputs(prevMemory, inputSequence)
        return memory, output

    output_sequence, _ = theano.scan(oneStep, sequences = [X], outputs_info = [memory_0, output_0], n_steps = sequenceLength)

    params = lstm.params

    params["W_out_px"] = weight_matrix((500, 1))
    params["b_out_px"] = weight_matrix((1,))

    p_X = T.nnet.sigmoid(T.dot(output_sequence[-1], params["W_out_px"]) + params["b_out_px"])

    return params, p_X

#Takes a matrix of noise variables, loads these into memory initially (concatenated with blank memory spots).  
#Outputs an X sequence.  
def generator(): 

    return params, X


sys.setrecursionlimit(100000)


if __name__ == "__main__": 

    print "RUN"

    weightSize = 60

    lstm = LSTMLayer.LSTMLayer(controllerSize = 60, memorySize = 60, outputSize = 60, inputSize = 60, initialScale = 0.05, useReluReadGate = True)

    W = theano.shared(numpy.random.normal(size = (weightSize,weightSize)))

    print W.shape

    def oneStep(prevH, prevMemory, prevController): 

        print "memory ndim should be 2", prevMemory.ndim

        controller1, memory1, h1 = lstm.getOutputs(prevController, prevMemory, prevH)

        return controller1, memory1, h1

    sequenceLength = 20

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

    print "output n dim should be 3", np.asarray(new_h_scan[0]).ndim
    print "new h scan", new_h_scan[0]

    #print "starting grad for loop"
    #timeStart = time.time()
    #g = T.grad(sum(map(T.sum, new_h)), lstm.params)
    #print "time spent on for loop grad", time.time() - timeStart

    timeStart = time.time()
    g_scan = T.grad(T.sum(new_h_scan), lstm.params.values())
    print "time spent on scan grad", time.time() - timeStart

    #timeStart = time.time()
    #f = theano.function(inputs = [h0, memory_0], outputs = [new_h[-1]] + g)
    #print "time spent compling for loop", time.time() - timeStart

    timeStart = time.time()
    f_scan = theano.function(inputs = [h0, memory_0, controller_0], outputs = [new_h_scan[0]])
    print "time spent compiling scan", time.time() - timeStart

    numIter = 20

    #timeStart = time.time()
    #for i in range(0, numIter): 
    #    f([[1.0] * weightSize] * batch_size, [[1.0] * weightSize] * batch_size)
    #print "time", time.time() - timeStart

    timeStart = time.time()

    for i in range(0, numIter): 
        output = f_scan([[1.0] * weightSize] * batch_size, [[1.0] * weightSize] * batch_size, [[1.0] * weightSize] * batch_size)

        print np.asarray(output[0]).shape

    print "time for scan version", time.time() - timeStart














