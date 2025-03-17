import sklearn.metrics as m
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.layers import *
from keras.models import Model
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, precision_score, \
    recall_score
from sklearn.metrics import roc_curve
from losses import WeightedBinaryCrossEntropy
from callbacks import KGCNMetric
from layers import CentralityEncoding, GraphormerBlock, AttentionFusion
from models.base_model import BaseModel
tf.random.set_seed(42)



class InterTREE(BaseModel):
    def __init__(self, config):
        super(InterTREE, self).__init__(config)

    def build(self):
        feature = Input(
            shape=(self.config.n_neighbors*self.config.n_neighbors*self.config.n_graphs*66,), name="input_node_feature", dtype='float'
        )

        feature_reshape = tf.reshape(feature,(-1,self.config.n_graphs, self.config.n_neighbors,self.config.n_neighbors,66))
        original_node_feature = feature_reshape[:,:,:,0,:64] ##[batch_size, n_graphs, n_neighbors,64]
        degree_node = Lambda(lambda x: K.expand_dims(x,axis=-1))(feature_reshape[:,:,:,0,64])  ##[batch_size, n_graphs, n_neighbors, 1]
        spatial_matrix = Lambda(lambda x: K.expand_dims(x,axis=2))(feature_reshape[:,:,:,:,65]) ##[batch_size, n_graphs, 1, n_neighbors, n_neighbors]

        print(original_node_feature.shape)  ##[-1,graphs,neighbors,64]
        print(degree_node.shape) ##[-1,graphs,neighbors]
        print(spatial_matrix.shape) ##[-1,n_graphs,neighbors,neighbors]

        centrEncodingLayer = CentralityEncoding(self.config.max_degree, self.config.d_model, name='centrality_encoding')
        graphormer_layers_common_channel = [
            GraphormerBlock(self.config.d_model, self.config.num_heads, self.config.dff, self.config.dropout,
                            self.config.d_sp_enc, self.config.sp_enc_activation, name=f'graphormer_{_}')
            for _ in range(self.config.n_layers)]

        node_embedding = []
        subgraph_results = []


        ##node_feature [batch_size, n_graphs, n_neighbors, d_model]
        ##distance [batch_size, n_graphs, n_neighbors, all_nodes]
        ##spatial [batch_size, n_graphs, multi-hop, n_neighbors, n_neighbors]

        for g in range(self.config.n_graphs):
            centr_encoding = centrEncodingLayer(degree_node[:, g, :, :])
            out = Lambda(lambda x: self.get_node_feature(x[0], x[1]), name=f'get_node_feature_{g}')(
                [original_node_feature[:, g, :, :], centr_encoding])
            out = Dense(self.config.d_model, activation=self.config.top_activation)(out)

            spatial_matrix_in_subgraphs = spatial_matrix[:, g, :, :,:]  ##spatial_matrix [batch_size, multi-hop, n_neighbors, n_neighbors]
            mask = Lambda(lambda x: self.create_padding_mask(x))(spatial_matrix_in_subgraphs[:, 0, :, :])
            attention_mask = mask[:, tf.newaxis, :, :]

            for n in range(self.config.n_layers):
                spatial_matrix_hop = spatial_matrix_in_subgraphs[:, 0, :, :] ##multihop [batch_size, n_neighbors, n_neighbors]
                attention_mask_n = attention_mask
                out, attn = graphormer_layers_common_channel[n](out, self.config.training, attention_mask_n,spatial_matrix_hop)  # (batch_size, inp_seq_len, d_model)

            node_embedding.append(out[:, 0, :])  ##save target node embeddings
            subgraph_results.append(out)

        aggreated_out = tf.keras.layers.Concatenate(name="concatenate_ouputs")(
            [node_embedding[i] for i in range(-len(node_embedding), 0)])  ##[bz, d_model * 3]


        attentionLayer = AttentionFusion(d_model=self.config.d_model, n_channels=self.config.n_graphs, name='attention_fusion')
        finalembedding, _ = attentionLayer(aggreated_out)


        outputs = Dense(1, activation='sigmoid', name='binary_classifier')(finalembedding)
        model = Model(feature, outputs)
        model.compile(optimizer=self.config.optimizer, loss=self.weight_loss, metrics=['acc'])

        return model

    def weight_loss(self, y_true, y_pred):

        l = WeightedBinaryCrossEntropy(self.config.loss_mul)

        return l.get_loss(y_true, y_pred)

    def create_padding_mask(self, nodes):
        return tf.cast(tf.math.equal(nodes, -1), tf.float32)

    def get_node_feature(self, node_embedding, centr_encoding):

        node_feature = node_embedding
        node_feature *= tf.math.sqrt(tf.cast(self.config.d_model, tf.float32))
        node_feature += centr_encoding  ##[batch_size, 1, embed_dim]

        return tf.keras.layers.Dropout(self.config.dropout)(node_feature, training=self.config.training)

    def add_metrics(self, x_train, y_train, x_val, y_val):
        self.callbacks.append(KGCNMetric(x_train, y_train, x_val, y_val, self.config.dataset))

    def fit(self, x_train, y_train, x_val, y_val):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_val, y_val)
        self.callbacks.append(ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5,
                                                verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))
        self.init_callbacks()

        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size = self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(x_val, y_val), callbacks=self.callbacks)
        print('Logging Info - training end...')

    def get_variables(self):
        return self.model.trainable_weights

    def predict(self, x):
        return self.model.predict(x).flatten()

    def score(self, x, y, threshold=0.5):  ##要重新实现
        y_true = y.flatten()
        y_pred = self.model.predict(x).flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_pred)
        p, r, t = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(r, p)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        p = precision_score(y_true=y_true, y_pred=y_pred)
        r = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        return auc, acc, p, r, f1, aupr, fpr.tolist(), tpr.tolist()
