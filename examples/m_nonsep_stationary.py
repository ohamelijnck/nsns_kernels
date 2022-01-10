import pickle
import matplotlib.pyplot as plt
from gpflow.decors import params_as_tensors

import sys
import numpy as np
import pandas as pd

import tensorflow as tf
import gpflow



import gpflow
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.models.gpr import GPR
from gpflow.training import AdamOptimizer, ScipyOptimizer

from scipy.cluster.vq import kmeans2
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

import os


class nonsep_RQ(gpflow.kernels.Kernel):
    def __init__(self, input_dim, active_dims,variance= 0.3):
        gpflow.kernels.Kernel.__init__(self, input_dim=input_dim, active_dims=active_dims)
        self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)
        a = np.array([0.05,0.2])
        self.lambda0 = gpflow.Param(0.05, transform=gpflow.transforms.positive)
        self.lambda1 = gpflow.Param(0.05, transform=gpflow.transforms.positive)
        self.lambda2 = gpflow.Param(0.05, transform=gpflow.transforms.positive)
        self.lengthscale = gpflow.Param(a, transform=gpflow.transforms.positive)
    @params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X       
        K = 1.0
        dist_1 = tf.reshape(X[:,0], (-1, 1)) - tf.reshape(X2[:, 0], (1, -1))
        dist_2 = tf.reshape(X[:,1], (-1, 1)) - tf.reshape(X2[:, 1], (1, -1))
        
        dist_1 = tf.square(dist_1)/(2*tf.square(self.lengthscale[0]))
        dist_2 = tf.square(dist_2)/(2*tf.square(self.lengthscale[1]))
        
        K1 = tf.pow(1+dist_1/self.lambda0+dist_2/self.lambda0,-self.lambda0)
        K2 = tf.pow(1+dist_1/self.lambda1,-self.lambda1)
        K3 = tf.pow(1+dist_2/self.lambda2,-self.lambda2)
            
        return self.variance*K1*K2*K3#tf.square(tf.reshape(X, (-1, 1)) - tf.reshape(X, (1, -1)))
    
    @params_as_tensors
    def Kdiag(self, X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))




def get_config(t='all'):
    configs = {}
    config_arr = []
    #number of datasets to try:
    for i in range(1):
        config_arr.append(
            {
                'file_prefix': 'split_gp_nonsep_st_{i}'.format(i=i),
                'ignore': False,
                'data_train': '../data/split_data_train_{i}.pickle'.format(i=i),
                'data_test': '../data/split_data_test_{i}.pickle'.format(i=i),
                'epochs': 5000,
                'train': True,
                'restore': False,
            }
        )
    configs['split'] = config_arr.copy()

    config_arr = []
    for i in range(9):
        config_arr.append(
            {
                'file_prefix': 'random_gp_nonsep_st_{i}_0'.format(i=i),
                'ignore': False,
                'data_train': '../data/random_data_train_{i}.pickle'.format(i=i),
                'data_test': '../data/random_data_test_{i}.pickle'.format(i=i),
                'epochs': 3000,
                'train': True,
                'restore': False,
            }
        )
    configs['random'] = config_arr.copy()

    if t == 'all': 
        return configs['random']+configs['split']
    return configs[t]

def main(CONFIG, return_m=False, force_restore=False):
   #===========================Load Data===========================
    
    data = pickle.load(open( CONFIG['data_train'], "rb" ))
    data_xs = pickle.load(open( CONFIG['data_test'], "rb" ))

    X = data['X']
    Y = data['Y']
    print('X: ', X.shape)
    print('Y: ', Y.shape)
    #===========================Create Model===========================

    k = nonsep_RQ(input_dim = 2,active_dims = [0,1])
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

    pickle.dump(results, open( "../results/{file}.pickle".format(file=CONFIG['file_prefix']), "wb" ) )
    pickle.dump(meta, open( "../results/{file}_meta.pickle".format(file=CONFIG['file_prefix']), "wb" ) )
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


