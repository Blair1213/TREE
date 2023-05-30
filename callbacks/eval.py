from collections import defaultdict

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score,precision_recall_curve, recall_score
import sklearn.metrics as m
from utils import write_log

#添加指标：ACC, AUPR, AUC-ROC, F1 +std

class KGCNMetric(Callback):
    def __init__(self, x_train, y_train, x_val, y_val, dataset):
        self.x_train = x_train
        self.y_val = y_val
        self.y_train = y_train
        self.x_val = x_val
        self.dataset=dataset
        self.threshold=0.5

        super(KGCNMetric, self).__init__()


    def on_epoch_end(self, epoch, logs=None):

        y_pred = self.model.predict(self.x_val).flatten()
        y_true = self.y_val.flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        precision, recall, _thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr=m.auc(recall,precision)
        y_pred = [1 if prob >= self.threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        logs['val_aupr']=float(aupr)
        logs['val_auc'] = float(auc)
        logs['val_acc'] = float(acc)
        logs['val_f1'] = float(f1)
        
        logs['dataset']=self.dataset
        logs['epoch_count']=epoch+1
        print(f'Logging Info - epoch: {epoch+1}, val_auc: {auc}, val_aupr: {aupr}, val_acc: {acc}, val_f1: {f1}')
        write_log('log/train_history.txt', logs, mode='a')

    @staticmethod
    def get_user_record(data, is_train):
        user_history_dict = defaultdict(set)
        for interaction in data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if is_train or label == 1:
                user_history_dict[user].add(item)
        return user_history_dict

