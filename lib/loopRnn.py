import theano
import theano.tensor as T
import numpy
import sys
import time
import LSTMLayer
import numpy as np
import Updates
import random
import qqplot
import matplotlib.pyplot as plt
from generator import generator
from discriminator import discriminator

cast32 = lambda x: T.cast(x, 'float32')

sys.setrecursionlimit(100000)

tmpDir = "tmp/"

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

    sequenceLength = 1
    batch_size = 100

    #Discriminator on observed sample, discriminator on generated sample

    params_disc, p_x_observed = discriminator(X_observed, sequenceLength, batch_size, params = None)
    params_gen, x_gen = generator(sequenceLength, batch_size, params = None)
    params_disc, p_x_generated = discriminator(x_gen, sequenceLength, batch_size, params = params_disc)

    loss_generator = adversarial_loss_generator(p_x_generated = p_x_generated)

    loss_discriminator = adversarial_loss_discriminator(p_x_generated = p_x_generated, p_x_observed = p_x_observed)

    learning_rate = 0.0001

    gen_updates = Updates.Updates(paramMap = params_gen, loss = loss_generator, learning_rate = learning_rate * 0.1)
    disc_updates = Updates.Updates(paramMap = params_disc, loss = loss_discriminator, learning_rate = learning_rate * 10.0)

    #Functions: 
    #Train discriminator and generator
    #Train only discriminator
    #Generate values without training

    print "disc update keys", len(disc_updates.getUpdates().keys())

    print "gen update keys", len(gen_updates.getUpdates().keys())

    print "joined update keys", len(dict(gen_updates.getUpdates().items() + disc_updates.getUpdates().items()))

    generate_sample = theano.function(inputs = [], outputs = [x_gen])
    trainDiscriminator = theano.function(inputs = [X_observed], updates = disc_updates.getUpdates())
    trainAll = theano.function(inputs = [X_observed], outputs = [loss_generator, loss_discriminator], updates = dict(gen_updates.getUpdates().items() + disc_updates.getUpdates().items()))

    g = generate_sample()

    print g[0].shape

    sample_prob = theano.function(inputs = [X_observed], outputs = [p_x_observed])

    sampled = sample_prob(g[0])

    from DataTransformation.plotData import getData

    p50Lst = []
    p90Lst = []

    for epoch in range(0, 400): 
        dataLst = getData()

        #seq_length x batch x 1

        for ts in dataLst: 
            if random.uniform(0,1) < 0.0: 
                trainDiscriminator(ts)
            else:         
                loss_gen, loss_disc = trainAll(ts)
                print "loss gen", loss_gen
                print "loss disc", loss_disc

            #print sample_prob(ts)
            print "===================="
            print ts[0][0]
            print "sample_prob", sample_prob(ts)[0][0].tolist()

        #Plot dist

        print "pulling samples for evaluation"

        allSamples = []

        for j in range(0, 16):
            samples = np.asarray(generate_sample()).flatten().tolist()
            allSamples += samples

        if random.uniform(0,1) < 1.0: 

            print sorted(allSamples)

            allTS = []

            for ts in dataLst[0:12]: 
                allTS += ts.flatten().tolist()

            binrange = np.arange(-2.0, 60.0, 1.0)
            p_data = []

            for val in binrange: 
                val_run = np.asarray([[[val]] * 100], dtype = 'float32')
                p_data += [sample_prob(val_run)[0][0][0]]

            plt.scatter(binrange, p_data)
            plt.hist(allTS, bins = binrange, normed = 1, alpha = 0.2)
            plt.hist(allSamples, bins = binrange, normed = 1, alpha = 0.2)

            plt.savefig(tmpDir + "hist_" + str(epoch) + ".png")
            plt.clf()

            #qqplot.qqplot(allSamples, allTS)

            print "mean samples", np.average(np.asarray(allSamples))
            print "mean observed", np.average(np.asarray(allTS))

            print "stdv samples", np.std(np.asarray(allSamples))
            print "stdv observed", np.std(np.asarray(allTS))

            print "p50 samples", np.percentile(np.asarray(allSamples), 50.0)
            print "p50 observed", np.percentile(np.asarray(allTS), 50.0)

            print "p90 samples", np.percentile(np.asarray(allSamples), 90.0)
            print "p90 observed", np.percentile(np.asarray(allTS), 90.0)

            p50Loss = abs(np.percentile(np.asarray(allSamples), 50.0) - np.percentile(np.asarray(allTS), 50.0))
            p90Loss = abs(np.percentile(np.asarray(allSamples), 90.0) - np.percentile(np.asarray(allTS), 90.0))

            p50Lst += [p50Loss]
            p90Lst += [p90Loss]

            plt.plot(p50Lst)
            plt.plot(p90Lst)

            plt.savefig(tmpDir + "progress_" + str(epoch) + ".png")
            plt.clf()

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





