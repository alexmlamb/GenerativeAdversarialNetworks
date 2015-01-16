import theano
import numpy as np
import theano.tensor as T

theano.config.floatX = 'float32'

def normalize(stepped_param):
    #using 4.0
    squared_filter_length_limit = 15.0
    col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
    desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
    scale = desired_norms / (1e-7 + col_norms)
    return stepped_param * scale


class Updates: 

    def __init__(self, paramMap, loss, learning_rate): 
        g_mom = {}
        updates = {}
        sqr_gradients = {}
        paramObjLst = paramMap.values()

        obj2Grad = {}

        l2_loss = 0.0

        for param in paramObjLst:
            gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX))
            g_mom[param] = gparam_mom

            sqr_grad = theano.shared(np.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX))

            sqr_gradients[param] = sqr_grad

            l2_loss += T.sum(param**2)

        gradLst = T.grad(loss, paramObjLst)

        for i in range(0, len(paramObjLst)): 
            obj2Grad[paramObjLst[i]] = gradLst[i]

        for param in paramObjLst:
            gparam = g_mom[param]
            sqr_grad = sqr_gradients[param]
            #new_gradient = T.grad(loss, param)

            new_gradient = obj2Grad[param]

            scaling_factor  = T.maximum(1.0, (T.sqrt(T.sum(T.sqr(new_gradient)))))
            #Divide by the norm of the gradient if it is greater than one
            new_gradient = new_gradient / scaling_factor

            new_gradient = T.switch(T.isnan(new_gradient), 0.0, new_gradient)

            mom = 0.7

            learning_rate_use = learning_rate / (T.sqrt(sqr_grad) + 1.0)

            updates[gparam] = T.cast(mom * gparam - (1.0 - mom) * learning_rate_use * new_gradient, theano.config.floatX)

            updates[sqr_grad] = T.cast(T.clip(sqr_grad + T.abs_(new_gradient), 0.0, 10000.0), theano.config.floatX)

        for param in paramObjLst:
            updated_value = param + updates[g_mom[param]]


            if param.ndim == 2:
                updated_value = normalize(updated_value)

            updates[param] = T.cast(updated_value, theano.config.floatX)

        self.updates = updates

    def getUpdates(self): 

        return self.updates






