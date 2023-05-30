# -*- coding: utf-8 -*-
import sys
import random
import os
import numpy as np
import h5py
import scipy.sparse as sp
from numpy.linalg import inv
import networkx as nx
from utils import Node2vec
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.getcwd())  # add the env path
from sklearn.model_selection import train_test_split, StratifiedKFold
from main import train


from config import H5_FILE, RESULT_LOG, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, SUBGRAPHA_TEMPLATE, SPATIAL_TEMPLATE, ModelConfig, ADJ_TEMPLATE, FEATURE_TEMPLATE, SHORT_PATH
from utils import pickle_dump, format_filename, write_log, pickle_load


def cross_validation_sets(y, mask, folds):
    label_idx = np.where(mask == 1)[0] # get indices of labeled genes
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=100)
    splits = kf.split(label_idx, y[label_idx])
    k_sets = []
    for train, test in splits:
        # get the indices in the real y and mask realm
        train_idx = label_idx[train]
        test_idx = label_idx[test]

        k_sets.append((train_idx, y[train_idx], test_idx, y[test_idx]))

        assert len(train_idx) == len(y[train_idx])
        assert len(test_idx) == len(y[test_idx])

    return k_sets

def read_h5file(path, network_name='network', feature_name='features'):
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        y_train = f['y_train'][:]
        y_test = f['y_test'][:]
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None
        train_mask = f['mask_train'][:]
        test_mask = f['mask_test'][:]
        if 'mask_val' in f:
            val_mask = f['mask_val'][:]
        else:
            val_mask = None

    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def create_subgraphs_randomwalk(dataset: str, adj, n_graphs: int, n_neighbors: int, is_distance = False):

    n_nodes = adj.shape[0]
    pmat = np.ones(shape=(n_nodes, n_nodes), dtype=int) * np.inf
    print(n_nodes)
    edge_index_begin, edge_index_end = np.where(adj > 0)
    print(len(edge_index_end))
    edge_index = np.array([edge_index_begin, edge_index_end]).transpose().tolist()
    my_graph = nx.Graph()
    my_graph.add_edges_from(tuple(edge_index))
    walks = Node2vec(graph=my_graph, path_length=n_neighbors, num_paths=n_graphs, workers=6, dw=True).get_walks() ##[n_nodes, n_graphs, n_neighbors]
    
    ##heterogeneous
    new_walks = np.zeros(shape=(n_nodes, n_graphs, n_neighbors), dtype=int)
    for i in range(walks.shape[0]):
        new_walks[walks[i][0][0], :, :] = walks[i]

    walks = new_walks

    if is_distance == False:
        path = nx.all_pairs_shortest_path_length(my_graph)
        for node_i, node_ij in path:
            if node_i % 1000 == 0:
                print(node_i)
            for node_j, length_ij in node_ij.items():
                pmat[node_i, node_j] = length_ij
        pmat[pmat == np.inf] = -1
        save_path = "sp/" + dataset + "_sp.h5"
        new_file = h5py.File(save_path, 'w')
        new_file.create_dataset(name="sp", shape=(n_nodes, n_nodes), data=pmat)
    else:
        f = h5py.File("sp/" + dataset + "_sp.h5")
        pmat = f["sp"][:]
        pmat[pmat == np.inf] = -1

    subgraphs_list = []
    for id in range(n_nodes):
        sub_subgraph_list = []
        for g in range(n_graphs):
            node_feature_id = np.array(walks[id, g, :],dtype=int)

            attn_bias = np.concatenate([np.expand_dims(i[node_feature_id, :][:, node_feature_id], 0) for i in [pmat]])

            sub_subgraph_list.append(attn_bias)

        subgraphs_list.append(sub_subgraph_list)

    return walks, np.array(subgraphs_list)

def process_data(dataset: str, n_graphs: int, n_neighbors: int, n_layers: int, lr: float, spatial: str, cv_folds: int, dropout: float, loss_mul: float, dff: int, bz: int, distance: bool):

    print("reading data.....")
    datapath = H5_FILE[dataset]
    print(datapath)
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = read_h5file(datapath)
    max_degree = int(max(np.sum(adj, axis=-1)) + 1)
    degree = np.expand_dims(np.sum(adj, axis=-1), axis=-1)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, ADJ_TEMPLATE, dataset=dataset), degree)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, FEATURE_TEMPLATE, dataset=dataset), features)

    neighbor_id = None
    spatial_matrix = None

    subgraph_path = format_filename(PROCESSED_DATA_DIR, SUBGRAPHA_TEMPLATE, dataset=dataset, strategy = spatial, n_channel = n_graphs, n_neighbor = n_neighbors)

    if os.path.exists(subgraph_path) == False:
        neighbor_id, spatial_matrix = create_subgraphs_randomwalk(dataset, adj, n_graphs, n_neighbors, distance)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, SUBGRAPHA_TEMPLATE, dataset=dataset, strategy=spatial,n_channel=n_graphs, n_neighbor=n_neighbors), neighbor_id)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, SPATIAL_TEMPLATE, dataset=dataset, strategy=spatial,n_channel=n_graphs,n_neighbor=n_neighbors), spatial_matrix)

    y_train_val = np.logical_or(y_train, y_val)
    mask_train_val = np.logical_or(train_mask, val_mask)
    k_sets = cross_validation_sets(y=y_train_val, mask=mask_train_val, folds=cv_folds) ##split training set and validation set


    test_id = np.where(test_mask == 1)[0]  # get indices of labeled genes
    y_test = y_test[test_id]

    temp = {'dataset': dataset, 'avg_auc': 0.0, 'avg_acc': 0.0, 'avg_aupr': 0.0, 'auc_std': 0.0, 'aupr_std': 0.0}
    results = {'auc':[], 'aupr': [], 'acc':[]}
    count = 0
    for i in range(cv_folds):
        train_id, y_train, val_id, y_val = k_sets[i]

        train_log = train(
            Kfold=i,
            dataset=dataset,
            train_label=y_train,
            test_label=y_test,
            val_label=y_val,
            train_id = train_id,
            test_id = test_id,
            val_id = val_id,
            n_graphs = n_graphs,
            n_neighbors = n_neighbors,
            n_layers = n_layers,
            spatial_type = spatial,
            max_degree=max_degree, 
            batch_size = bz,
            embed_dim = 64,
            num_heads = 4,
            d_sp_enc = dff,
            dff = dff,
            l2_weights=5e-7,
            lr=lr,
            dropout = dropout,
            loss_mul= loss_mul,
            optimizer ='adam',
            n_epoch=100,
            callbacks_to_add=['modelcheckpoint', 'earlystopping']
        )

        count += 1
        results['auc'].append(train_log['test_auc'])
        results['acc'].append(train_log['test_acc'])
        results['aupr'].append(train_log['test_aupr'])

    temp['avg_acc'] = np.mean(np.array(results['acc']))
    temp['avg_auc'] = np.mean(np.array(results['auc']))
    temp['avg_aupr'] = np.mean(np.array(results['aupr']))
    temp['auc_std'] = np.std(np.array(results['auc']))
    temp['aupr_std'] = np.std(np.array(results['aupr']))

    write_log(format_filename(LOG_DIR, RESULT_LOG[dataset]), temp, 'a')
    print(f'Logging Info - {cv_folds} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]},'
          f'avg_aupr: {temp["avg_aupr"]}, auc_std: {temp["auc_std"]}, aupr_std: {temp["aupr_std"]}')



if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    model_config = ModelConfig()

    #process_data(dataset: str, n_graphs: int, n_neighbors: int, n_layers: int, lr: float, spatial: str, cv_folds: int)

    process_data('CPDB', 6, 8, 3, 0.001, "rw", 10, dropout=0.5, loss_mul=0.27, dff=128, bz=32,distance=True)
    process_data('STRINGdb', 3, 8, 5, 0.004, "rw", 10, dropout=0.5, loss_mul=0.24, dff=256, bz=64, distance=True)
    process_data('PCNET', 5, 10, 2, 0.004, 'rw', 10, dropout=0.1, loss_mul=0.14, dff=128, bz=64,distance=True)
    process_data('IREF', 5, 10, 3, 0.006, "rw", 10, dropout=0.5, loss_mul=0.18, dff=128, bz=64, distance=True)
    process_data('IREF_2015', 5, 10, 3, 0.006, "rw", 10, dropout=0.4, loss_mul=0.29, dff=128, bz=64, distance=True)
    process_data('MULTINET', 5, 8, 6, 0.003, "rw", 10, dropout=0.5, loss_mul=0.18, dff=128, bz=64, distance = True)
    process_data('LTG', 10, 14, 8, 0.003, "rw", 10, dropout=0.5, loss_mul=0.2, dff=128, bz=64, distance=True)
    process_data('MTG', 10, 14, 4, 0.004, "rw", 10, dropout=0.5, loss_mul=0.16, dff=128, bz=64, distance=True)



