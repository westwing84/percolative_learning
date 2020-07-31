# 浸透学習の実装

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
import numpy as np
import matplotlib.pyplot as plt
from function import network, MNISTDataset, get_main_aux_data, Trainer, make_tensorboard, shuffle_pixel, shuffle_datasets, LossAccHistory

maindt_size = 784       # 主データのサイズ
subdt_size = 784        # 補助データのサイズ
shuffle_rate = 0.5      # 主データのシャッフル率
layers_percnet = 2      # 浸透サブネットの層数
layers_intnet = 3       # 統合サブネットの層数
percnet_size = 100      # 浸透サブネットの各層の素子数
percfeature_size = 100  # 浸透特徴の個数
intnet_size = 100       # 統合サブネットの各層の素子数
output_size = 10        # 出力データのサイズ
epochs_prior = 200      # 事前学習のエポック数
epochs_perc = 1000      # 浸透学習のエポック数
epochs_adj = 300        # 微調整のエポック数
batch_size = 1024       # バッチサイズ
validation_split = 1 / 7  # 評価に用いるデータの割合
test_split = 1 / 7        # テストに用いるデータの割合
verbose = 2             # 学習進捗の表示モード
decay = 0.05            # 減衰率
optimizer = Adam(lr=0.0001)      # 最適化アルゴリズム
# callbacks = [make_tensorboard(set_dir_name='log')]  # コールバック


# ニューラルネットワークの構成
percnet, network = network((maindt_size+subdt_size,),
                           percfeature_size, output_size,
                           layers_percnet, layers_intnet,
                           percnet_size, intnet_size)

percnet.compile(optimizer=optimizer, loss=mean_squared_error)
network.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])


# MNISTデータの読み込み
datasets = MNISTDataset()
x_train, y_train, x_test, y_test = datasets.get_data()
# TrainデータとTestデータを結合して，それをTrain，Validation，Testデータに分ける．
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)
# x, y = shuffle_datasets(x, y)
x_train, y_train, x_val, y_val, x_test, y_test = get_main_aux_data(x, y, validation_split, test_split, shuffle_rate)
# ValidationとTestデータの補助データは全て0にする
x_val[:, :subdt_size] = 0
x_test[:, :subdt_size] = 0

'''
# 主データの作成
x_train_main = shuffle_pixel(x_train_aux, shuffle_rate)
x_test_main = shuffle_pixel(x_test_aux, shuffle_rate_test)
x_test_aux = 0 * x_test_aux    # テストデータの補助データはすべて0(主データのみで再現できるか確認するため)
# 主データと補助データを結合
x_train = np.concatenate([x_train_aux, x_train_main], axis=1)
x_test = np.concatenate([x_test_aux, x_test_main], axis=1)
validation_id = int(validation_split * x_train.shape[0])
x_val = x_train[-validation_id:]
x_train = x_train[:-validation_id]
y_val = y_train[-validation_id:]
y_train = y_train[:-validation_id]
x_val[:, :subdt_size] = 0
'''

'''
# 入力データの表示
n = 10
plt.figure()
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i][:subdt_size].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(x_test[i][subdt_size:].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

# 学習
trainer = Trainer(percnet, network, optimizer, verbose)
history_list = LossAccHistory()
history_list = trainer.train(x_train, y_train,
                             x_val, y_val,
                             subdt_size,
                             batch_size,
                             epochs_prior, epochs_perc, epochs_adj,
                             decay,
                             history_list)

# 損失と精度の評価
score_train = network.evaluate(x_train, y_train, batch_size=batch_size)
score_val = network.evaluate(x_val, y_val, batch_size=batch_size)
score_test = network.evaluate(x_test, y_test, batch_size=batch_size)
print('Train - loss:', score_train[0], '- accuracy:', score_train[1])
print('Validation - loss:', score_val[0], '- accuracy:', score_val[1])
print('Test - loss:', score_test[0], '- accuracy:', score_test[1])

# 損失と精度をグラフにプロット
plt.figure()
plt.plot(history_list.accuracy)
plt.plot(history_list.accuracy_val)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.0, 1.01)
plt.legend(['Train', 'Validation'])
plt.show()

plt.figure()
plt.plot(history_list.losses)
plt.plot(history_list.losses_val)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


