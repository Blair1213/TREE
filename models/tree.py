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
from keras.regularizers import l2
tf.random.set_seed(42)



class TREE(BaseModel):
    def __init__(self, config):
        super(TREE, self).__init__(config)

    def build(self):
        input_node_id = Input(
            shape=(1,), name="input_node_id", dtype='int64'
        )  ##[n_nodes, d_model]


        centrEncodingLayer = CentralityEncoding(self.config.max_degree, self.config.d_model, name='centrality_encoding')
        graphormer_layers_common_channel = [
            GraphormerBlock(self.config.d_model, self.config.num_heads, self.config.dff, self.config.dropout,
                            self.config.d_sp_enc, self.config.sp_enc_activation, name=f'graphormer_{_}')
            for _ in range(self.config.n_layers)]

        node_embedding = []
        attention_each_layer = []

        ##node_feature [batch_size, n_graphs, n_neighbors, d_model]
        ##distance [batch_size, n_graphs, n_neighbors, all_nodes]
        ##spatial [batch_size, n_graphs, multi-hop, n_neighbors, n_neighbors]

        sub_node_feature, sub_distance, sub_spatial = Lambda(lambda x: self.get_sub_info(x), name='get_sub_info')(
            input_node_id)

        embedding_out_per_layer = []
        node_embedding_all_layers = []

        for g in range(self.config.n_graphs):

            centr_encoding = centrEncodingLayer(sub_distance[:,g,:,:])
            out = Lambda(lambda x: self.get_node_feature(x[0],x[1]),name=f'get_node_feature_{g}')([sub_node_feature[:,g,:,:], centr_encoding])
            out = Dense(self.config.d_model, activation=self.config.top_activation)(out)

            spatial_matrix_in_subgraphs = sub_spatial[:, g, :, :, :]  ##spatial_matrix [batch_size, multi-hop, n_neighbors, n_neighbors]
            mask = Lambda(lambda x: self.create_padding_mask(x))(spatial_matrix_in_subgraphs[:,0,:,:])
            attention_mask = mask[:, tf.newaxis, :, :]

            for n in range(self.config.n_layers):
                spatial_matrix_hop = spatial_matrix_in_subgraphs[:, 0, :, :]  ##[batch_size, n_neighbors, n_neighbors]
                attention_mask_n = attention_mask
                out, attn = graphormer_layers_common_channel[n](out, self.config.training, attention_mask_n, spatial_matrix_hop)  # (batch_size, inp_seq_len, d_model)
                ##attn_shape[num_heads, neighbors, neighbors]
                attention_each_layer.append(attn)

            node_embedding.append(out[:, 0, :])

        aggreated_out = tf.keras.layers.Concatenate(name="concatenate_ouputs")(
            [node_embedding[i] for i in range(-len(node_embedding), 0)])  ##[bz, d_model * 3]

        #attn_out = tf.keras.layers.Concatenate(name='attention_outputs')(
        #    [attention_each_layer[i] for i in range(-len(attention_each_layer), 0)]
        #)

        attentionLayer = AttentionFusion(d_model=self.config.d_model, n_channels=self.config.n_graphs, name='attention_fusion')
        finalembedding, _ = attentionLayer(aggreated_out)  ##[batch_size, d_model]

        outputs = Dense(1, activation='sigmoid', name='binary_classifier')(finalembedding)
        model = Model(input_node_id, outputs)
        model.compile(optimizer=self.config.optimizer, loss=self.weight_loss, metrics=['acc'])

        return model


    def get_final_embedding_all(self, attention_weights, embeddings):

        embeddings = tf.reshape(embeddings,(-1, self.config.n_graphs, self.config.n_layers, self.config.d_model))
        embeddings = tf.transpose(embeddings,(0,2,1,3)) ##[bz,n_layers,n_graphs,d_model]
        attention_weights = tf.expand_dims(tf.reshape(attention_weights,(-1,1,self.config.n_graphs)),axis=1)##[bz,1,1,n_graphs]
        attention_weights = tf.repeat(attention_weights,axis=1,repeats=self.config.n_layers) ##[bz,n_layers,1,n_graphs]

        return tf.matmul(attention_weights,embeddings)

    def weight_loss(self, y_true, y_pred):

        l = WeightedBinaryCrossEntropy(self.config.loss_mul)

        return l.get_loss(y_true, y_pred)

    def get_sub_info(self, node_id):

        node_neighbors = tf.squeeze(tf.gather(self.config.node_neighbor, node_id),1) ##[batch_size, n_graphs, n_neighbors]
        node_feature = tf.gather(self.config.node_feature, node_neighbors) ##[batch_size, n_graphs, n_neighbors, d_model]
        distance = tf.gather(self.config.distance_matrix, node_neighbors) ##[batch_size, n_graphs, n_neighbors, all_nodes]
        spatial = tf.squeeze(tf.gather(self.config.spatial_matrix, node_id),1) ##[batch_size, n_graphs, multi-hop, n_neighbors, n_neighbors]

        return node_feature, distance, spatial;


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
