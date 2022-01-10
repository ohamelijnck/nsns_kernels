import logging, os
import numpy as np
import tensorflow as tf

#disable TF warnings
if True:
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
    tf.logging.set_verbosity(tf.logging.ERROR)

import pickle
import matplotlib.pyplot as plt
from gpflow.decors import params_as_tensors

import sys
import pandas as pd

import gpflow
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.models.gpr import GPR
from gpflow.training import AdamOptimizer, ScipyOptimizer

from scipy.cluster.vq import kmeans2
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

import os


class ns_RBF(gpflow.kernels.Kernel):
    def __init__(self, input_dim, active_dims,variance= 1.0, lengthscales=[0.1, 0.2]):
        gpflow.kernels.Kernel.__init__(self, input_dim=input_dim, active_dims=active_dims)
        self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)
        a = lengthscales[0]*np.ones(input_dim)
        b = lengthscales[1]*np.ones(input_dim)
        self.lengthscale_a = gpflow.Param(a, transform=gpflow.transforms.positive)
        self.lengthscale_b = gpflow.Param(b, transform=gpflow.transforms.positive)
        
    def hyper_function(self, X, input_dim):
        return self.lengthscale_b[input_dim] + self.lengthscale_a[input_dim]*X
    
    @params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
            
        K = 1
        for i in range(self.input_dim):
            Xi = X[:,i]
            Xi2 = X2[:, i]
            dist = tf.reshape(Xi, (-1, 1)) - tf.reshape(Xi2, (1, -1))
            
            lengthscale = tf.abs(self.hyper_function(Xi,i))+0.001
            lengthscale2 = tf.abs(self.hyper_function(Xi2,i))+0.001


            dist2 = tf.reshape(lengthscale, (-1, 1)) + tf.reshape(lengthscale2, (1, -1))
            sq = tf.matmul( tf.expand_dims(lengthscale,1),  tf.expand_dims(lengthscale2,1), transpose_b=True)
            K = K*tf.sqrt(2*tf.sqrt(sq)/dist2)*tf.exp(-2*tf.square(dist)/dist2)
        return self.variance*K#tf.square(tf.reshape(X, (-1, 1)) - tf.reshape(X, (1, -1)))
    
    @params_as_tensors
    def Kdiag(self, X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


def get_config(t='all'):
    configs = {}
    config_arr = []
    lengthscales = [[0.1, 0.2]]
    for ls_i, ls in enumerate(lengthscales):
        for i in range(1):
            config_arr.append(
                {
                    'file_prefix': 'split_gp_sep_nonst_{ls_i}_{i}'.format(i=i, ls_i=ls_i),
                    'data_train': '../data/split_data_train_{i}.pickle'.format(i=i),
                    'data_test': '../data/split_data_test_{i}.pickle'.format(i=i),
                    'ignore': False,
                    'epochs': 5000,
                    'lengthscale': ls,
                    'restore': False,
                    'train': True,
                }
            )
    configs['split'] = config_arr.copy()
    config_arr = []

    lengthscales = [[1, 2]]
    for ls_i, ls in enumerate(lengthscales):
        for i in range(9):
            config_arr.append(
                {
                    'file_prefix': 'random_gp_sep_nonst_0_{i}_{ls_i}'.format(i=i, ls_i=ls_i),
                    'data_train': '../data/random_data_train_{i}.pickle'.format(i=i),
                    'data_test': '../data/random_data_test_{i}.pickle'.format(i=i),
                    'ignore': False,
                    'epochs': 10000,
                    'lengthscale': ls,
                    'restore': False,
                    'train': True,
                }
            )
    configs['random'] = config_arr.copy()

    if t == 'all': 
        return configs['random']+configs['split']
    return configs[t]

def main(CONFIG, return_m=False, force_restore=False):
    refresh = 100
    #===========================Load Data===========================
    data = pickle.load(open( CONFIG['data_train'], "rb" ))
    data_xs = pickle.load(open( CONFIG['data_test'], "rb" ))

    X = data['X']
    Y = data['Y']

    print('X: ', X.shape)
    print('Y: ', Y.shape)

    #===========================Load Model===========================

    #set to true to use RBF kernel
    debug = False

    if debug:
        k = gpflow.kernels.RBF(2) #for debugging
    else:
        k = ns_RBF(2,[0,1], lengthscales=CONFIG['lengthscale'])

    m = gpflow.models.GPR(X, Y, kern=k)
    m.compile()
    tf_session = m.enquire_session()

    m.likelihood.variance.trainable = True
    m.likelihood.variance = 0.01
    #===========================Optimise===========================

    if CONFIG['restore']:
        saver = tf.train.Saver()
        saver.restore(tf_session, 'restore/{name}.ckpt'.format(name=CONFIG['file_prefix']))

    elbos = []

    def logger(x):
        refresh=100
        sess = m.enquire_session()
        obj = m.objective.eval(session=sess)
        elbos.append(obj)
        if x % refresh == 0:
            print(obj)

    if CONFIG['train']:
        opt = AdamOptimizer(0.01)
        opt.minimize(m, step_callback=logger,  maxiter=config['epochs'])

        saver = tf.train.Saver()
        save_path = saver.save(tf_session, "restore/{name}.ckpt".format(name=CONFIG['file_prefix']))


    #===========================Predict and store results===========================
    def predict_fn(x):
        return m.predict_y(x)

    results = {}
    for src in data_xs:
        ys, ys_var = predict_fn(data_xs[src]['X'])
        results[src] = {}
        results[src]['mean'] = ys
        results[src]['var'] = ys_var

    meta = {
        'elbos': elbos
    }

    #pickle.dump(results, open( "../results/{file}.pickle".format(file=CONFIG['file_prefix']), "wb" ) )
    #pickle.dump(meta, open( "../results/{file}_meta.pickle".format(file=CONFIG['file_prefix']), "wb" ) )
    print(m.as_pandas_table())
    print('Finished')



if __name__ == '__main__':
    i = 7
    t ='random'
    if len(sys.argv) == 2:
        i = int(sys.argv[1])

    if len(sys.argv) == 3:
        t = sys.argv[2]


    configs = get_config(t)
    config = configs[i]
    main(config)



