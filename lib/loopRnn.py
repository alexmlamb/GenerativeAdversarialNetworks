import theano
import theano.tensor as T
import numpy
import sys
import time
import LSTMLayer
import numpy as np
import Updates
import random

print "run"

'''

Build discriminator, generator.  Create sample sequences from multivariate normal distribution with linear trend.  Also consider using fresh demand?  First train to generate sequence without conditioning on part of sequence?  

'''

cast32 = lambda x: T.cast(x, 'float32')

def weight_matrix(shape): 
    random_seed = 1235
    rng = np.random.RandomState(random_seed)
    scale = 0.05
    return theano.shared(numpy.asarray(scale * rng.normal(size = shape), dtype = theano.config.floatX))

#Provide two methods.  One gets a network that takes an X sequence and outputs p(X).  
def discriminator(X, sequenceLength, batch_size, params = None): 

    memorySize = 200
    outputSize = 200

    lstm = LSTMLayer.LSTMLayer(inputSize = 1, controllerSize = 500, memorySize = memorySize, outputSize = outputSize, initialScale = 0.05, params = params)

    memory_0 = theano.shared(np.zeros(shape = (batch_size, memorySize), dtype = 'float32'))
    output_0 = theano.shared(np.zeros(shape = (batch_size, outputSize), dtype = 'float32'))

    def oneStep(inputSequence, prevMemory, prevOutput):
        memory, output = lstm.getOutputs(prevMemory, inputSequence)
        return cast32(memory), cast32(output)

    output_vars, _ = theano.scan(oneStep, sequences = [X], outputs_info = [memory_0, output_0], n_steps = sequenceLength)

    final_memory, final_output = output_vars

    if params == None: 
        params = lstm.params
        params["W_out_px"] = weight_matrix((outputSize, 1))
        params["b_out_px"] = weight_matrix((1,))

    p_X = T.nnet.sigmoid(T.dot(final_output[-1], params["W_out_px"]) + params["b_out_px"])

    return params, p_X

def generate_noise(size, batch_size): 
    random_seed = 1234
    rng = np.random.RandomState(random_seed)
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    num_comp = 5
    comp_size = size / num_comp

    z1 = srng.normal(size = (batch_size, comp_size))
    z2 = srng.binomial(size = (batch_size,comp_size), n = 1, p = 0.5, dtype = 'float32')
    z3 = srng.multinomial(size = (batch_size,), n = 1, pvals = [1.0 / (comp_size)] * comp_size, dtype = 'float32')
    z4 = srng.multinomial(size = (batch_size,), n = 1, pvals = [1.0 / (comp_size)] * comp_size, dtype = 'float32')
    z5 = srng.multinomial(size = (batch_size,), n = 1, pvals = [1.0 / (comp_size)] * (comp_size), dtype = 'float32')

    z = T.concatenate([z1,z2,z3,z4,z5], axis = 1)

    return z

#Takes a matrix of noise variables, loads these into memory initially (concatenated with blank memory spots).  
#Outputs an X sequence.  
def generator(sequenceLength, batch_size, params): 

    size_noise_memory = 50
    size_other_memory = 350

    z = cast32(generate_noise(size_noise_memory, batch_size))

    other_memory = theano.shared(np.zeros(shape = (batch_size, size_other_memory), dtype = 'float32'), name = "other_memory")    

    initial_memory = T.concatenate([z, other_memory], axis = 1)

    memorySize = size_noise_memory + size_other_memory
    outputSize = 1
    inputSize = 1

    output_0 = theano.shared(np.zeros(shape = (batch_size, outputSize), dtype = 'float32'), name = "output_tensor")

    lstm = LSTMLayer.LSTMLayer(inputSize = inputSize, controllerSize = 200, memorySize = memorySize, outputSize = outputSize, initialScale = 0.05, params = params)

    def oneStep(prevMemory, prevOutput):
        inputSequence = np.zeros(shape = (batch_size, inputSize), dtype = 'float32')
        memory, output = lstm.getOutputs(prevMemory, inputSequence)
        return cast32(memory), cast32(output)

    output_vars, _ = theano.scan(oneStep, sequences = [], outputs_info = [initial_memory, output_0], n_steps = sequenceLength)

    final_memory, final_output = output_vars

    params = lstm.params

    return params, final_output


sys.setrecursionlimit(100000)

def adversarial_loss_generator(p_x_generated): 
    return 1.0 * T.sum(T.log(1.0 - p_x_generated))

def adversarial_loss_discriminator(p_x_generated, p_x_observed): 
    return 1.0 * (-1.0 * T.sum(T.log(p_x_observed)) - 1.0 * T.sum(T.log(1.0 - p_x_generated)))


def test_gan_rnn(): 

    #Set up parameters

    #The time series for each element of the batch
    #sequenceLength x batch_size x 1
    X_observed = T.tensor3()

    #Each sequence is either 1 or 0.  
    Y = T.vector()

    sequenceLength = 10
    batch_size = 100

    #Discriminator on observed sample, discriminator on generated sample

    params_disc, p_x_observed = discriminator(X_observed, sequenceLength, batch_size, params = None)
    params_gen, x_gen = generator(sequenceLength, batch_size, params = None)
    params_disc, p_x_generated = discriminator(x_gen, sequenceLength, batch_size, params = params_disc)

    loss_generator = adversarial_loss_generator(p_x_generated = p_x_generated)

    loss_discriminator = adversarial_loss_discriminator(p_x_generated = p_x_generated, p_x_observed = p_x_observed)

    learning_rate = 0.1

    gen_updates = Updates.Updates(paramMap = params_gen, loss = loss_generator, learning_rate = learning_rate)
    disc_updates = Updates.Updates(paramMap = params_disc, loss = loss_discriminator, learning_rate = learning_rate)

    #Functions: 
    #Train discriminator and generator
    #Train only discriminator
    #Generate values without training

    print "disc update keys", len(disc_updates.getUpdates().keys())

    print "gen update keys", len(gen_updates.getUpdates().keys())

    print "joined update keys", len(dict(gen_updates.getUpdates().items() + disc_updates.getUpdates().items()))

    generate_sample = theano.function(inputs = [], outputs = [x_gen])
    trainDiscriminator = theano.function(inputs = [X_observed], updates = disc_updates.getUpdates())
    trainAll = theano.function(inputs = [X_observed], outputs = [loss_generator, loss_discriminator], updates = gen_updates.getUpdates())

    g = generate_sample()

    print g[0].shape

    sample_prob = theano.function(inputs = [X_observed], outputs = [p_x_observed])

    sampled = sample_prob(g[0])

    from DataTransformation.plotData import getData


    for epoch in range(0, 200): 
        dataLst = getData()

        #seq_length x batch x 1

        for ts in dataLst: 
            if random.uniform(0,1) < 0.9: 
                trainDiscriminator(ts)
            else:         
                loss_gen, loss_disc = trainAll(ts)
                print "loss gen", loss_gen
                print "loss disc", loss_disc

            #print sample_prob(ts)
            #print "sample", np.asarray(generate_sample()[0]).swapaxes(0,1)[0].tolist()
            print "===================="
            print ts[0][0]
            print "sample_prob", sample_prob(ts)[0][0].tolist()
            print "sample", np.asarray(generate_sample()[0]).swapaxes(0,1)[0].tolist()
    


#Generate samples from two different distributions and see if discriminator can learn to separate them
def test_discriminator_1(): 
    #The time series for each element of the batch
    #sequenceLength x batch_size x 1
    X = T.tensor3()
    #Each sequence is either 1 or 0.  
    Y = T.vector()

    sequenceLength = 20
    batch_size = 100

    params_disc, p_x = discriminator(X, sequenceLength, batch_size)

    

if __name__ == "__main__": 
    test_gan_rnn()





