import sys
import theano
import theano.tensor as T
import random
import numpy as np
import matplotlib.pyplot as plt
import Updates
import qqplot
import time

theano.config.floatX = 'float32'


#3 elements.  
#Discriminator on true point
#Discriminator on generated point (these share the same parameters)
#Generator

poolSize = 5
maxout = lambda vector: T.max(T.reshape(vector, (vector.shape[0], vector.shape[1] / poolSize, poolSize)), axis = 2)
relu = lambda vector: T.maximum(0.0, vector)
activation = maxout


n = 20000
td1 = np.random.gamma(1.0,2.0, n / 2)
td2 = np.random.normal(-3.0,2.0, n / 2)
true_dist = td1.tolist() + td2.tolist()
random.shuffle(true_dist)
true_dist = np.asarray(true_dist)
#true_dist = np.random.binomial(1, 0.5, n)

mean = true_dist.mean()
stdv = np.sqrt(true_dist.var())

#Have a tentative inverse CDF.  Generate a random point z ~ U(0,1).  
#Classifier that determines if point is real or from the generator.  
#

#Generator network maximizes log(D(G(Z)))
#This means maximizing the probability that a random point from the generator is classified as a true point.  

discriminator_params = {}
generator_params = {}

random_seed = 1234
rng = np.random.RandomState(random_seed)
srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

#using 400/1200/10
num_hidden_discriminator = 400
num_hidden_generator = 1200
var_dimensionality = 200

#using 0.01
scale_disc = 0.05
scale_gen = 0.05

castx = lambda x: T.cast(x, theano.config.floatX)

discriminator_params["W1_d"] = theano.shared(np.asarray(1.0 * np.random.uniform(-1.0 * scale_disc, 1.0 * scale_disc, (1, num_hidden_discriminator * 5)), dtype = theano.config.floatX))
discriminator_params["b1_d"] = theano.shared(np.asarray(0.0 + 0.0 * np.random.normal(0, 1, (5 * num_hidden_discriminator,)), dtype = theano.config.floatX))

discriminator_params["W2_d"] = theano.shared(np.asarray(1.0 * np.random.uniform(-1.0 * scale_disc, 1.0 * scale_disc, (num_hidden_discriminator, num_hidden_discriminator * 5)), dtype = theano.config.floatX))
discriminator_params["b2_d"] = theano.shared(np.asarray(0.0 + 0.0 * np.random.normal(0, 0.1, (5 * num_hidden_discriminator,)), dtype = theano.config.floatX))

discriminator_params["W3_d"] = theano.shared(np.asarray(1.0 * np.random.uniform(-1.0 * scale_disc, 1.0 * scale_disc, (num_hidden_discriminator, 1)), dtype = theano.config.floatX))
discriminator_params["b3_d"] = theano.shared(np.asarray(0.0 * np.random.normal(0, 0.1, (1,)), dtype = theano.config.floatX))

generator_params["W1_g"] = theano.shared(np.asarray(1.0 * np.random.uniform(-1.0 * scale_gen, 1.0 * scale_gen, (var_dimensionality, num_hidden_generator)), dtype = theano.config.floatX), name = "W1_g")
generator_params["b1_g"] = theano.shared(np.asarray(0.0 + 0.0 * np.random.normal(0, 1, (num_hidden_generator,)), dtype = theano.config.floatX))

generator_params["W2_g"] = theano.shared(np.asarray(1.0 * np.random.uniform(-1.0 * scale_gen, 1.0 * scale_gen, (num_hidden_generator, num_hidden_generator)), dtype = theano.config.floatX))
generator_params["b2_g"] = theano.shared(np.asarray(0.0 + 0.0 * np.random.normal(0, 1, (num_hidden_generator,)), dtype = theano.config.floatX))

generator_params["W3_g"] = theano.shared(np.asarray(1.0 * np.random.uniform(-1.0 * scale_gen, 1.0 * scale_gen, (num_hidden_generator, 1)), dtype = theano.config.floatX))
generator_params["b3_g"] = theano.shared(np.asarray(0.0 * np.random.normal(0, 1, (1,)), dtype = theano.config.floatX))

#To try: 
# -Different forms for input z.  Try giving both z and erfinv(z).  

#z is length 1 vector
#W1 is 1 by h_len matrix
#b2 is length 1 vector
#W2 is h_len by 1 matrix
#Returns G(z).  
#h = max(0.0, W1*z + b1)
#x = W2 * h + b2
def generator_network(z, params): 
    #z = T.erfinv(z)
    #using tanh

    h1 = relu(T.dot(z, params["W1_g"]) + params["b1_g"])

    #h1 *= srng.binomial(n=1, p=0.5, size=h1.shape)

    h2 = relu(T.dot(T.concatenate([h1], axis = 1), params["W2_g"]) + params["b2_g"])

    #h2 *= srng.binomial(n=1, p=0.5, size=h2.shape)

    x = T.dot(T.concatenate([h2], axis = 1), params["W3_g"]) + params["b3_g"]

    return x

#Discriminator maximizes log(1 - D(G(Z))) + log(D(X))
#Minimizing probability that a random point from the generator is classified as a true point.  
#Maximizing probability that true points are classified as true points.  

#x is length 1 vector
#Returns D(x)
def discriminator_network(x, params, trainMode): 
    #x = (x - mean) / stdv
    h1 = activation(T.dot(x, params["W1_d"]) + params["b1_d"])

    if trainMode: 
        h1 *= castx(srng.binomial(n=1, p=0.5, size=h1.shape))
    else: 
        h1 *= 0.5

    h2 = activation(T.dot(h1, params["W2_d"]) + params["b2_d"])

    if trainMode: 
        h2 *= castx(srng.binomial(n=1, p=0.5, size=h2.shape))
    else: 
        h2 *= 0.5

    y = T.dot(T.concatenate([h2], axis = 1), params["W3_d"]) + params["b3_d"]
    return T.nnet.sigmoid(y)


learning_rate = T.scalar()
x = T.matrix()
#z = T.matrix()

#z = srng.normal(avg = 0,std = 1, size = (100, var_dimensionality))
z2 = srng.binomial(size = (100,var_dimensionality / 4), n = 1, p = 0.5, dtype = 'float32')
z3 = srng.multinomial(size = (100,), n = 1, pvals = [1.0 / (var_dimensionality / 4)] * (var_dimensionality / 4), dtype = 'float32')
z4 = srng.multinomial(size = (100,), n = 1, pvals = [1.0 / (var_dimensionality / 4)] * (var_dimensionality / 4), dtype = 'float32')
z5 = srng.multinomial(size = (100,), n = 1, pvals = [1.0 / (var_dimensionality / 4)] * (var_dimensionality / 4), dtype = 'float32')

z = T.concatenate([z2,z3,z4,z5], axis = 1)

#z = T.erfinv(z)

#Value between 0 and 1 corresponding to the probability that a point belongs to the true data distribution
discriminator_true_value = discriminator_network(x, discriminator_params, trainMode = True)
discriminator_true_value_test = discriminator_network(x, discriminator_params, trainMode = False)

#Generated value, intended to mimic true data distribution.  
generator = generator_network(z, generator_params)


discriminator_sample = discriminator_network(generator, discriminator_params, trainMode = True)

#these losses are minimized
generation_loss = 1.0 * T.sum(T.log(1.0 - discriminator_sample))
discriminator_loss = 1.0 * (-1.0 * T.sum(T.log(discriminator_true_value)) - 1.0 * T.sum(T.log(1.0 - discriminator_sample)))

generation_updates = Updates.Updates(paramMap = generator_params, loss = generation_loss, learning_rate = learning_rate)
discriminator_updates = Updates.Updates(paramMap = discriminator_params, loss = discriminator_loss, learning_rate = learning_rate)


train_discriminator = theano.function([x, learning_rate], outputs = [discriminator_sample, discriminator_loss], updates = discriminator_updates.getUpdates())
#train_generator = theano.function([z, learning_rate], outputs = [generator, discriminator_sample, generation_loss], updates = generation_updates.getUpdates())

train = theano.function([x,learning_rate], outputs = [generator, discriminator_sample, generation_loss, discriminator_loss], updates = dict(discriminator_updates.getUpdates().items() + generation_updates.getUpdates().items()))

test_discriminator = theano.function([x], outputs = [discriminator_true_value_test])
test_generator = theano.function([], outputs = [generator])

numIterations = 400

initial_learning_rate = 0.1

learning_rate = np.asarray(initial_learning_rate, dtype = theano.config.floatX)

loss_lst = []

for i in range(0, numIterations): 

    totalDLoss = 0.0
    totalGLoss = 0.0

    true_val_lst = []
    rand_cdf_lst = []

    t_start = time.time()

    for j in range(0, n): 


        true_value = true_dist[j]
        random_cdf = np.random.uniform(0,1,var_dimensionality)

        true_val_lst += [[true_value]]
        rand_cdf_lst += [random_cdf]

        #using 20
        if len(true_val_lst) == 100: 

            true_val_lst = np.asarray(true_val_lst, dtype = theano.config.floatX)
            rand_cdf_lst = np.asarray(rand_cdf_lst, dtype = theano.config.floatX)
            
            #disc_sample, dloss = train_discriminator(true_val_lst, rand_cdf_lst, learning_rate)
            #generated_value, p_gen_val_true, gloss = train_generator(rand_cdf_lst, learning_rate)

            #was using 0.95
            if random.uniform(0,1) < 0.9:
                disc_sample, dloss = train_discriminator(true_val_lst, learning_rate)
                totalDLoss += dloss
            else: 
                generated_value, disc_sample, gloss, dloss = train(true_val_lst, learning_rate)
                totalDLoss += dloss
                totalGLoss += gloss

            true_val_lst = []
            rand_cdf_lst = []

    print "Time in epoch", time.time() - t_start

    print "total D loss", totalDLoss
    print "total G loss", totalGLoss

    val_points = []
    p_d = []
    dist_p50 = []
    for k in range(-10, 10): 
        test_val = np.asarray([[k]], dtype = theano.config.floatX)
        print test_val, test_discriminator(test_val)
        val_points += [test_val[0][0]]
        p_d += [test_discriminator(test_val)[0][0].tolist()[0]]
        dist_p50 += [abs(0.5 - p_d[len(p_d) - 1])]

    #print val_points, p_d

    print "average p(d)", sum(dist_p50) * 1.0 / len(dist_p50)

    loss_lst += [sum(dist_p50) * 1.0 / len(dist_p50)]

    #print "=======FORMALEVAL=========================="
    #for k in range(1, 20): 
    #    kr = k * 1.0 / 20
    #    true_q = np.percentile(true_dist, kr * 100.0)
    #    predicted_q, disc, gl = train_generator(np.asarray([[kr]], dtype = theano.config.floatX), 0.0)
        #print kr, predicted_q, true_q, disc

    #only plot after 50 iter
    #if i != numIterations - 1: 
    #    continue

    #Plot every 5 epochs
    if i % 15 != 0: 
        continue

    #plot samples
    samples = []
    for k in range(0, 5000):
        random_cdf = np.random.uniform(0,1,var_dimensionality)
        predicted_q = test_generator()[0]
        samples.append(predicted_q[0][0])

    print max(samples)

    binrange = np.arange(-10.0, 10.0, 0.5)

    plt.scatter(val_points, p_d)
    plt.hist(samples, bins = binrange, normed = 1, alpha = 0.2)
    plt.hist(true_dist, bins = binrange, normed = 1, alpha = 0.2)
    plt.show(block=True)

    qqplot.qqplot(samples, true_dist)

#Evaluate the network

    plt.plot(loss_lst)
    plt.show(block=True)

