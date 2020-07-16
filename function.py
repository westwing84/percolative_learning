import os
import random
import numpy as np
from time import gmtime, strftime
from tensorflow.keras.callbacks import TensorBoard, Callback


# Tensorboardの作成
def make_tensorboard(set_dir_name=''):
    tictoc = strftime('%a_%d_%b_%Y_%H_%M_%S', gmtime())
    directory_name = tictoc
    log_dir = set_dir_name + '_' + directory_name
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir)
    return tensorboard


# lossおよびAccuracyを保存するためのクラス
class LossAccHistory(Callback):
    def __init__(self):
        self.losses = []
        self.accuracy = []
        self.losses_val = []
        self.accuracy_val = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.losses_val.append(logs.get('val_loss'))
        self.accuracy_val.append(logs.get('val_accuracy'))


# データのうちshuffle_rate(0~1)の割合のものをシャッフルする関数
def shuffle_data(data, shuffle_rate):
    dtsize = data.shape[1]
    dtnum_shuffled = int(dtsize * shuffle_rate)
    id_shuffled = []
    i = 0
    while i < dtnum_shuffled:
        rnd = random.randint(0, dtsize - 1)
        if not rnd in id_shuffled:
            id_shuffled.append(rnd)
            i += 1
    id_shuffled = np.array(id_shuffled)
    id = id_shuffled
    id_shuffled = np.random.permutation(id_shuffled)
    idlist = np.arange(dtsize)
    for j in range(dtnum_shuffled):
        idlist[id[j]] = idlist[id_shuffled[j]]
    data = data[:, idlist]
    return data

def concat(dt1, dt2):
    dt = np.concatenate([dt1, dt2], axis=1)
    return dt

