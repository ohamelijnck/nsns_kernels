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
import gpflow
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.models.gpr import GPR
from gpflow.training import AdamOptimizer, ScipyOptimizer

import sys
import pandas as pd
from scipy.cluster.vq import kmeans2


def get_config(t='all'):
    configs = {}
    config_arr = []
    lengthscales = [0.01]
    for ls_i, ls in enumerate(lengthscales):
        for i in range(1):
            config_arr.append(
                {
                    'file_prefix': 'split_dgp_100_{ls_i}_{i}'.format(i=i, ls_i=ls_i),
                    'data_train': '../data/split_data_train_{i}.pickle'.format(i=i),
                    'data_test': '../data/split_data_test_{i}.pickle'.format(i=i),
                    'ignore': False,
                    'epochs': 50000,
                    'num_inducing': None,
                    'lengthscale': ls,
                    'restore': False,
                    'train': True,
                }
            )

    configs['split'] = config_arr.copy()

    config_arr = []
    lengthscales = [0.01]
    for ls_i, ls in enumerate(lengthscales):
        for i in range(7):
            config_arr.append(
                {
                    'file_prefix': 'random_dgp_100_0_{i}_{ls_i}'.format(i=i, ls_i=ls_i),
                    'data_train': '../data/random_data_train_{i}.pickle'.format(i=i),
                    'data_test': '../data/random_data_test_{i}.pickle'.format(i=i),
                    'ignore': False,
                    'epochs': 50000,
                    'num_inducing': None,
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
    #@TODO: hack to get working on cluster
    from doubly_stochastic_dgp.dgp import DGP

    #===========================Load Data===========================
    data = pickle.load(open( CONFIG['data_train'], "rb" ))
    data_xs = pickle.load(open( CONFIG['data_test'], "rb" ))

    X = data['X']
    Y = data['Y']

    #===========================Remove NaNs===========================
    X_raw = X.copy()
    Y_raw = Y.copy()

    idx = (~np.isnan(Y[:, 0]))
    X = X[idx, :] 
    Y = Y[idx, :] 

    num_z = CONFIG['num_inducing']
    if num_z is None:
        Z = X
    else:
        Z = kmeans2(X, num_z, minit='points')[0] 

    print('X: ', X.shape)
    print('Y: ', Y.shape)
    print('Z: ', Z.shape)

    #===========================Load Model===========================

    #@TODO make starting parameters the same
    #@TODO in 

    def make_DGP(L, X, Y, Z):
        kernels = []
        input_dims = [2, 1]
        kernels = [
            gpflow.kernels.RBF(2, lengthscales=CONFIG['lengthscale'], variance=1.) ,
            gpflow.kernels.RBF(1, lengthscales=CONFIG['lengthscale'], variance=1.) 
            #gpflow.kernels.Polynomial(input_dim=3, degree=1.0) + White(1, variance=1e-5)
        ]

        m_dgp = DGP(X, Y, Z, kernels, Gaussian(variance=0.01), num_samples=1)
        
        # init the layers to near determinisic 
        for layer in m_dgp.layers[:-1]:
            layer.q_sqrt = layer.q_sqrt.value * 1e-5
        return m_dgp


    m_dgp = make_DGP(2, X, Y, Z)
    tf_session = m_dgp.enquire_session()

    #===========================Optimise===========================

    if CONFIG['restore']:
        saver = tf.train.Saver()
        saver.restore(tf_session, 'restore/{name}.ckpt'.format(name=CONFIG['file_prefix']))

    elbos = []
    def logger(x):
        refresh=100
        sess = m_dgp.enquire_session()
        obj = m_dgp.objective.eval(session=sess)
        elbos.append(obj)
        if x % refresh == 0:
            print(obj)


    if CONFIG['train']:
        opt = AdamOptimizer(0.01)
        opt.minimize(m_dgp, step_callback=logger,  maxiter=config['epochs'])

        saver = tf.train.Saver()
        save_path = saver.save(tf_session, "restore/{name}.ckpt".format(name=CONFIG['file_prefix']))


    #===========================Predict and store results===========================
    def predict_fn(x):
        ms, vs = m_dgp.predict_y(x, 1000)

        m = np.average(ms, 0)
        v = np.average(vs + ms**2, 0) - m**2

        return m, v

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




