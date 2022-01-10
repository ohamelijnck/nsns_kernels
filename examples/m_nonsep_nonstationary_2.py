import pickle
import matplotlib.pyplot as plt
from gpflow.decors import params_as_tensors

import sys
import numpy as np
import pandas as pd

import tensorflow as tf


import gpflow
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.models.gpr import GPR
from gpflow.training import AdamOptimizer, ScipyOptimizer

from scipy.cluster.vq import kmeans2
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

import os


class nsns_RQ2(gpflow.kernels.Kernel):
    def __init__(self, input_dim, active_dims,variance= 0.5):
        gpflow.kernels.Kernel.__init__(self, input_dim=input_dim, active_dims=active_dims)
        self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)
        a = 0.1*np.ones(input_dim)
        b = 0.1*np.ones(input_dim)
        self.lambda0 = gpflow.Param(0.1, transform=gpflow.transforms.positive)
        self.lambda1 = gpflow.Param(3.0 , transform=gpflow.transforms.positive)
        self.lambda2 = gpflow.Param(1.0, transform=gpflow.transforms.positive)
        self.lengthscale_a = gpflow.Param(a, transform=gpflow.transforms.positive)
        self.lengthscale_b = gpflow.Param(b, transform=gpflow.transforms.positive)
        self.lengthscale = gpflow.Param([50.0,50.0], transform=gpflow.transforms.positive)
        self.local_structure = gpflow.Param([0.1,1.0], transform=gpflow.transforms.positive)
    def hyper_function(self, X, input_dim):
        return self.lengthscale_b[input_dim] + self.lengthscale_a[input_dim]*X

    @params_as_tensors
    def K(self, X, X2=None):
        sim_num =100
        if X2 is None:
            X2 = X       
        K = 1.0
        M1 = tf.cast(tf.linspace(-1.5,1.5,sim_num),dtype=tf.dtypes.float64)
        M2 = tf.cast(tf.linspace(-1.5,1.5,sim_num),dtype=tf.dtypes.float64)
        #A = tf.reduce_min(X[:,2])
        #B = tf.reduce_max(X[:,2])
        #M3 = tf.linspace(1.5*A-0.5*B,1.5*B-0.5*A,100)   
        
        
        l = tf.size(X2[:,0],out_type=tf.dtypes.int32)
        l2 = tf.size(X[:,0],out_type=tf.dtypes.int32)
        const = [1,l]
        const2 = [1,l2]
        
        a1= tf.tile(tf.expand_dims(X[:,0],1),const)
        a12 = tf.transpose(tf.tile(tf.expand_dims(X2[:,0],1),const2))
        a2 = tf.tile(tf.expand_dims(X[:,1],1),const)
        a22 = tf.transpose(tf.tile(tf.expand_dims(X2[:,1],1),const2))
        #a3 = tf.tile(tf.expand_dims(X[:,2],1),const)
        #a32 = tf.transpose(tf.tile(tf.expand_dims(X2[:,2],1),const))
        
        d1 = tf.abs(tf.reshape(X[:,0], (-1, 1)) - tf.reshape(X2[:, 0], (1, -1)))
        d2 = tf.abs(tf.reshape(X[:,1], (-1, 1)) - tf.reshape(X2[:, 1], (1, -1)))
        #d3 = tf.abs(tf.reshape(X[:,2], (-1, 1)) - tf.reshape(X2[:, 2], (1, -1)))
        
        temp = 0.0
        for i in range(sim_num-1):
            k1 = tf.multiply(tf.exp( - 0.5 * tf.square((a1 - M1[i])/self.lengthscale[0])),tf.exp( - 0.5 * tf.square((a12 - M1[i])/self.lengthscale[0])))
            k2 = tf.multiply(tf.exp( - 0.5 * tf.square((a2 - M2[i])/self.lengthscale[1])),tf.exp( - 0.5 * tf.square((a22 - M2[i])/self.lengthscale[1])))
            #k3 = tf.multiply(tf.exp( - 0.5 * tf.square((a3 - M3[i])/self.lengthscale[2])),tf.exp( - 0.5 * tf.square((a32 - M3[i])/self.lengthscale[2])))
                
            Co1 = tf.pow(d1/(self.local_structure[0]),2.0)
            Co1 = 1+tf.abs(Co1)
            Co2 = tf.pow(d2/(self.local_structure[1]),2.0)
            Co2 = 1+tf.abs(Co2)
            #Co3 = tf.pow(d3/(self.local_structure[2]),2.0)
            #Co3 = 1+tf.abs(Co3)  
            
            Co = tf.multiply(tf.multiply(tf.pow(Co1+Co2-1,-self.lambda0-10+self.lambda1*((1.0+M1[i])*(1.0+M2[i]))),tf.pow(Co1,-self.lambda1*tf.exp((1.0+M1[i])))),tf.pow(Co2,-self.lambda1*tf.exp((1.0+M2[i]))))
            #Co = tf.multiply(Co,tf.pow(Co3,-1))
            
            #k = tf.multiply(tf.multiply(tf.multiply(k1,k2),k3),Co)
            k = tf.multiply(tf.multiply(tf.multiply(k1,k2),1),Co)
            temp = temp+tf.sqrt((M1[i+1]-M1[i])**2+(M2[i+1]-M2[i])**2)*k
        
        return self.variance*temp#tf.square(tf.reshape(X, (-1, 1)) - tf.reshape(X, (1, -1)))
    
    @params_as_tensors
    def Kdiag(self, X):
        return tf.diag_part(self.K(X))
        #return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))



def get_config(t='all'):
    configs = {}
    config_arr = []
    #number of datasets to try:
    for i in range(1):
        config_arr.append(
            {
                'file_prefix': 'split_gp_nonsep_nonst2_{i}'.format(i=i),
                'ignore': False,
                'data_train': '../data/split_data_train_{i}.pickle'.format(i=i),
                'data_test': '../data/split_data_test_{i}.pickle'.format(i=i),
                'epochs': 1,
                'train': True,
                'restore': False,
            }
        )
    configs['split'] = config_arr.copy()

    config_arr = []

    for i in range(7):
        config_arr.append(
            {
                'file_prefix': 'random_gp_nonsep_nonst2_{i}_0'.format(i=i),
                'ignore': False,
                'data_train': '../data/random_data_train_{i}.pickle'.format(i=i),
                'data_test': '../data/random_data_test_{i}.pickle'.format(i=i),
                'epochs': 1,
                'train': True,
                'restore': False,
            }
        )
    configs['random'] = config_arr.copy()


    if t == 'all': 
        return configs['random']+configs['split']
    return configs[t]

def main(CONFIG, return_m=False, force_restore=False):
    from util import batch_predict
   #===========================Load Data===========================
    
    data = pickle.load(open( CONFIG['data_train'], "rb" ))
    data_xs = pickle.load(open( CONFIG['data_test'], "rb" ))

    X = data['X']
    Y = data['Y']

    #===========================Create Model===========================

    k = nsns_RQ2(input_dim = 2,active_dims = [0,1])
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
        ys, ys_var = batch_predict(data_xs[src]['X'], predict_fn)
        results[src] = {}
        results[src]['mean'] = ys
        results[src]['var'] = ys_var

    meta = {
        'elbos': elbos
    }

    pickle.dump(results, open( "../results/{file}.pickle".format(file=CONFIG['file_prefix']), "wb" ) )
    pickle.dump(meta, open( "../results/{file}_meta.pickle".format(file=CONFIG['file_prefix']), "wb" ) )
    print('Finished')


if __name__ == '__main__':
    i = 0
    t ='all'
    if len(sys.argv) == 2:
        i = int(sys.argv[1])

    if len(sys.argv) == 3:
        t = sys.argv[2]


    configs = get_config(t)
    config = configs[i]
    main(config)


