# -*- coding: utf-8 -*-

import os

RAW_DATA_DIR = os.getcwd()+'/dataset/networks/'
VAR_DATA_DIR = os.getcwd()+'/mtg_variant/'
LVAR_DATA_DIR = os.getcwd()+'/ltg_variant/'
PROCESSED_DATA_DIR = os.getcwd()+'/pdata'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'
dataset = ['CPDB', 'PCNET', 'MULTINET', 'IREF', 'IREF_2015', 'STRINGdb', 'LTG', 'MTG']
H5_FILE = {
    'CPDB':os.path.join(RAW_DATA_DIR,'CPDB_multiomics.h5'),
    'PCNET':os.path.join(RAW_DATA_DIR,'PCNET_multiomics.h5'),
    'MULTINET':os.path.join(RAW_DATA_DIR,'MULTINET_multiomics.h5'),
    'IREF':os.path.join(RAW_DATA_DIR,'IREF_multiomics.h5'),
    'IREF_2015':os.path.join(RAW_DATA_DIR,'IREF_2015_multiomics.h5'),
    'STRINGdb':os.path.join(RAW_DATA_DIR,'STRINGdb_multiomics.h5'),
    'MTG':os.path.join(RAW_DATA_DIR,'MTG_multiomics.h5'),
    'LTG':os.path.join(RAW_DATA_DIR,'LTG_multiomics.h5'),
}

#
SPATIAL_TEMPLATE = '{dataset}_method_{strategy}_channel_{n_channel}_neighbor_{n_neighbor}_spatial.npy'
SUBGRAPHA_TEMPLATE = '{dataset}_method_{strategy}_channel_{n_channel}_neighbor_{n_neighbor}_subgraphs.npy'
ADJ_TEMPLATE = '{dataset}_adj.npy'
SHORT_PATH = '{dataset}_sp.npy'
FEATURE_TEMPLATE = '{dataset}_feature.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
RESULT_LOG={'STRINGdb':'STRINGdb_result.txt', 'IREF_2015': 'IREF_2015_result.txt', 'CPDB': 'CPDB_result.txt',
            'IREF':'IREF_result.txt', 'PCNET':'PCNET_result.txt', 'MULTINET':'MULTINET_result.txt', 'MTG':'MTG_result.txt','LTG':'LTG_result.txt'}

PERFORMANCE_LOG = 'TREE_performance'


class ModelConfig(object):
    def __init__(self):

        self.n_layers = 2
        self.d_model= 64
        self.l2_weight = 1e-7  # l2 regularizer weight
        self.lr = 0.005  # learning rate
        self.n_epoch = 50
        self.dff = 128
        self.max_degree = 1322
        self.n_neighbors = 3
        self.n_graphs = 3
        self.num_heads = 4
        self.concat_n_layers = self.n_layers
        self.dropout = 0.5
        self.loss_mul = 0.1
        self.d_sp_enc = 64
        self.batch_size = 64
        self.top_dropout = 0.5
        self.d_top = 256
        self.optimizer = 'adam'
        self.model_head = 'average'
        self.sp_enc_activation = "relu"
        self.top_activation = "relu"


        self.distance_matrix = None
        self.node_feature = None
        self.node_neighbor = None
        self.spatial_matrix = None
        self.training = None


        self.exp_name = None
        self.model_name = None
        
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 20
        self.early_stopping_verbose = 1
        self.K_Fold=1
        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3
