import theano
import theano.tensor as T
import numpy
import numpy.random as rng
import math

from ConsiderConstant import consider_constant

'''
    Implements a modified version of the long short term memory unit

    http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf

    The essential idea of the LSTM network is that there is a memory module which stores values that do not have to decay or change over time.  
    In forecasting, there are many things that the model might wish to store in the memory module.  The module may store the size of the last Christmas
    peak, it may store the number of promotions previously seen for an ASIN, or it may store some measure of price elasticity.  Simple experiments
    prove that the model is able to remember the exact magnitude of previous spikes.  

    The network accomplishes this by having a set of gates which explicitly control whether the value from a memory unit is read, whether there should
    be any writes into the memory module, and if the value of the module should decay over time.  

    This implementation does not use peephole connections as they made results significantly worse on both real and synthetic datasets.  
    One possible explanation is that the memory modules are storing demand for previous dates, so their values tend to be much larger than what is seen
    in the input features, which has the effect of making learning from the features more difficult

'''

class LSTMLayer: 

    def __init__(self, inputSize, controllerSize, memorySize, outputSize, initialScale, useReluReadGate = True):

        readGateSize = memorySize
        readDeltaSize = memorySize
        writeGateSize = memorySize
        keepGateSize = memorySize

        scale = initialScale

        #It is possible for the model to immediately enter a local minimum at the start of training in which the write gate or keep gate are closed for too
        #many memory modules, which prevents the model from learning long-range dependencies.  
        keepGateInitialBias = 5.0
        writeGateInitialBias = 2.0

        #All rectified linear unit hidden layers have this initial scale
        reluScale = 0.01

        self.useReluReadGate = useReluReadGate

        if useReluReadGate: 
            readGateScale = reluScale
        else:
            readGateScale = initialScale

        self.W_controller_0 = theano.shared(numpy.asarray(scale * rng.normal(size = (inputSize + 1 * memorySize, controllerSize)), dtype = theano.config.floatX), name = "Controller weights 0")

        self.W_controller_1 = theano.shared(numpy.asarray(scale * rng.normal(size = (controllerSize, controllerSize)), dtype = theano.config.floatX), name = "Controller weights 1")
        self.W_controller = theano.shared(numpy.asarray(scale * rng.normal(size = (controllerSize, controllerSize)), dtype = theano.config.floatX), name = "Controller weights 2")

        self.W_readgate = theano.shared(numpy.asarray(readGateScale * rng.normal(size = (readGateSize, inputSize + 1 * controllerSize)), dtype = theano.config.floatX), name = "read gate weights")

        self.W_readdelta = theano.shared(numpy.asarray(scale * rng.normal(size = (readDeltaSize, inputSize + 1 * controllerSize)), dtype = theano.config.floatX), name = "readdelta weights")

        self.W_writegate = theano.shared(numpy.asarray(scale * rng.normal(size = (writeGateSize, inputSize + 1 * memorySize)), dtype = theano.config.floatX), name = "writegate weights")

        self.W_keepgate = theano.shared(numpy.asarray(scale * rng.normal(size = (keepGateSize, inputSize + 1 * memorySize)), dtype = theano.config.floatX), name = "keepgate weights")

        self.W_output = theano.shared(numpy.asarray(reluScale * rng.normal(size = (outputSize, inputSize + memorySize + controllerSize)), dtype = theano.config.floatX), name = "output weights")

        self.b_controller_0 = theano.shared(numpy.asarray(numpy.zeros(shape = controllerSize), dtype = theano.config.floatX))
        self.b_controller_1 = theano.shared(numpy.asarray(numpy.zeros(shape = controllerSize), dtype = theano.config.floatX))
        self.b_controller = theano.shared(numpy.asarray(numpy.zeros(shape = controllerSize), dtype = theano.config.floatX))

        self.b_readgate = theano.shared(numpy.asarray(numpy.zeros(shape = readGateSize), dtype = theano.config.floatX))

        self.b_readdelta = theano.shared(numpy.asarray(numpy.zeros(shape = readDeltaSize), dtype = theano.config.floatX))

        self.b_writegate = theano.shared(numpy.asarray(writeGateInitialBias + numpy.zeros(shape = writeGateSize), dtype = theano.config.floatX))

        self.b_keepgate = theano.shared(numpy.asarray(numpy.zeros(shape = keepGateSize) + keepGateInitialBias, dtype = theano.config.floatX))

        self.b_output = theano.shared(numpy.asarray(numpy.zeros(shape = outputSize), dtype = theano.config.floatX))

        self.params = [self.W_controller, self.W_readgate, self.W_readdelta, self.W_writegate, self.W_keepgate, self.W_output, self.b_controller, self.b_readgate, self.b_readdelta, self.b_writegate, self.b_keepgate, self.b_output]

    '''
        Returns new controller, new memory
    '''
    def getOutputs(self, previousController, previousMemory, input_layer): 

        assert(previousController.ndim == previousMemory.ndim)
        assert(previousMemory.ndim == input_layer.ndim)

        if previousController.ndim == 1: 
            axisConcat = 0
        else:
            axisConcat = 1

        controller_0 = T.maximum(0.0, T.dot(T.concatenate([consider_constant(previousMemory), 1.0 * input_layer], axis = axisConcat), self.W_controller_0) + self.b_controller_0)        

        controller_1 = T.maximum(0.0, T.dot(controller_0, self.W_controller_1) + self.b_controller_1)

        controller = T.maximum(0.0, T.dot(controller_1, self.W_controller) + self.b_controller)

        #Have multiple layers in controller?  This determines what gets passed in / out from the network.  

        if self.useReluReadGate: 
            readgate = T.maximum(0.0, (T.dot(T.concatenate([controller, input_layer], axis = axisConcat), self.W_readgate.T) + self.b_readgate))
        else:
            readgate = T.nnet.sigmoid(T.dot(T.concatenate([controller, input_layer], axis = axisConcat), self.W_readgate.T) + self.b_readgate)

        readdelta = T.tanh(T.dot(T.concatenate([controller, input_layer], axis = axisConcat), self.W_readdelta.T) + self.b_readdelta)

        keepgate = T.nnet.sigmoid(T.dot(T.concatenate([controller, input_layer], axis = axisConcat), self.W_keepgate.T) + self.b_keepgate)

        memory_intermediate = previousMemory * keepgate + readgate * readdelta

        writegate = T.nnet.sigmoid(T.dot(T.concatenate([controller, input_layer], axis = axisConcat), self.W_writegate.T) + self.b_writegate)

        memory = memory_intermediate

        output = writegate * T.maximum(0.0, T.dot(T.concatenate([controller, memory, input_layer], axis = axisConcat), self.W_output.T) + self.b_output)

        return controller, memory_intermediate, output




