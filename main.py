# -*- coding: utf-8 -*-

import os
import gc
import time

import numpy as np
from keras import backend as K
from keras import optimizers
import tensorflow as tf

from utils import format_filename, write_log
from models import TREE
from config import ModelConfig, LOG_DIR, PERFORMANCE_LOG, PROCESSED_DATA_DIR, ADJ_TEMPLATE, FEATURE_TEMPLATE, SPATIAL_TEMPLATE, SUBGRAPHA_TEMPLATE
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def seed_tensorflow(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return tf.keras.optimizers.Adam(learning_rate)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))


def train(Kfold, dataset, train_label, test_label, val_label, train_id, test_id, val_id, n_graphs, n_neighbors, n_layers, spatial_type,
          max_degree, batch_size, embed_dim, num_heads, d_sp_enc, dff, l2_weights, lr, dropout, loss_mul, optimizer, n_epoch, callbacks_to_add=None, overwrite=True):

    config = ModelConfig()
    config.d_model = embed_dim
    config.n_layers = n_layers
    config.concat_n_layers = n_layers
    config.l2_weight = l2_weights
    config.dataset=dataset
    config.K_Fold=Kfold
    config.lr = lr
    config.batch_size = batch_size
    config.optimizer = get_optimizer(optimizer, lr)
    config.n_epoch = n_epoch
    config.max_degree = max_degree
    config.num_heads = num_heads
    config.n_graphs = n_graphs
    config.n_neighbors = n_neighbors
    config.dropout = dropout
    config.training = True
    config.d_sp_enc = d_sp_enc
    config.dff = dff
    config.loss_mul = loss_mul
    config.callbacks_to_add = callbacks_to_add

    config.distance_matrix = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_TEMPLATE, dataset=dataset), allow_pickle=True)
    config.node_feature = np.load(format_filename(PROCESSED_DATA_DIR, FEATURE_TEMPLATE, dataset=dataset), allow_pickle=True)
    config.node_neighbor = np.load(format_filename(PROCESSED_DATA_DIR, SUBGRAPHA_TEMPLATE, dataset=dataset, strategy = 'rw', n_channel = n_graphs,n_neighbor = n_neighbors), allow_pickle=True)
    config.spatial_matrix = np.load(format_filename(PROCESSED_DATA_DIR, SPATIAL_TEMPLATE, dataset=dataset, strategy = spatial_type, n_channel = n_graphs,n_neighbor = n_neighbors), allow_pickle=True)

    config.exp_name = f'TREE_{dataset}_spatialType_{spatial_type}_layerNums_{n_layers}_graphsNum_{n_graphs}_neighborsNum_{n_neighbors}_optimizer_{optimizer}_lr_{lr}__epoch_{n_epoch}'

    print(config.callbacks_to_add)
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str

    train_log = {'exp_name': config.exp_name, 'optimizer': optimizer,'epoch': n_epoch, 'learning_rate': lr,
                 'n_graphs':n_graphs, 'n_neighbors':n_neighbors}
    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = TREE(config)
    train_label=np.array(train_label)
    valid_label=np.array(val_label)
    test_label=np.array(test_label)

    train_id = np.array(train_id)
    valid_id = np.array(val_id)
    test_id = np.array(test_id)

    seed_tensorflow()
    #model.summary()
    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.fit(x_train = train_id, y_train=train_label, x_val = valid_id, y_val = valid_label)
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    print('Logging Info - Evaluate over valid data:')
    model.load_best_model()
    auc, acc, spe, sen, f1, aupr, fpr, tpr = model.score(x=valid_id, y=valid_label)


    print(f'Logging Info - dev_auc: {auc}, dev_acc: {acc}, dev_f1: {f1}, dev_aupr: {aupr}'
          )
    train_log['dev_auc'] = auc
    train_log['dev_acc'] = acc
    train_log['dev_sen'] = sen
    train_log['dev_spe'] = spe
    train_log['dev_f1'] = f1
    train_log['dev_aupr'] = aupr
    train_log['k_fold'] = Kfold
    train_log['dataset'] = dataset
    train_log['dev_fpr'] = fpr
    train_log['dev_tpr'] = tpr


    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over valid data based on swa model:')
        auc, acc, sen, spe, f1, aupr, fpr, tpr = model.score(x=valid_id, y=valid_label)

        train_log['swa_dev_auc'] = auc
        train_log['swa_dev_acc'] = acc
        train_log['swa_dev_sen'] = sen
        train_log['swa_dev_spe'] = spe
        train_log['swa_dev_f1'] = f1
        train_log['swa_dev_aupr'] = aupr
        train_log['swa_dev_fpr'] = fpr
        train_log['swa_dev_tpr'] = tpr
        #train_log['swa_dev_r'] = r
        #train_log['swa_dev_p'] = p
        print(f'Logging Info - swa_dev_auc: {auc}, swa_dev_acc: {acc}, swa_dev_f1: {f1}, swa_dev_aupr: {aupr}') #修改输出指标
    print('Logging Info - Evaluate over test data:')
    model.load_best_model()
    auc, acc, spe, sen, f1, aupr, fpr, tpr = model.score(x=test_id, y=test_label)

    train_log['test_auc'] = auc
    train_log['test_acc'] = acc
    train_log['test_sen'] = sen
    train_log['test_spe'] = spe
    train_log['test_f1'] = f1
    train_log['test_aupr'] = aupr
    train_log['test_fpr'] = fpr
    train_log['test_tpr'] = tpr
    #train_log['test_r'] = r
    #train_log['test_p'] = p
    print(f'Logging Info - test_auc: {auc}, test_acc: {acc}, test_f1: {f1}, test_aupr: {aupr}' )
    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over test data based on swa model:')
        auc, acc, spe, sen, f1, aupr, fpr, tpr = model.score(x=test_id, y=test_label)
        train_log['swa_test_auc'] = auc
        train_log['swa_test_acc'] = acc
        train_log['swa_test_sen'] = sen
        train_log['swa_test_spe'] = spe
        train_log['swa_test_f1'] = f1
        train_log['swa_test_aupr'] = aupr
        train_log['swa_test_fpr'] = fpr
        train_log['swa_test_tpr'] = tpr
        #train_log['swa_test_r'] = r
        #train_log['swa_test_p'] = p
        print(f'Logging Info - swa_test_auc: {auc}, swa_test_acc: {acc}, swa_test_f1: {f1}, swa_test_aupr: {aupr}')
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    del model
    gc.collect()
    K.clear_session()
    return train_log

