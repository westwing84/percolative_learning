# 浸透学習の実装

from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from function import make_tensorboard, shuffle_data, LossAccHistory

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
validation_split = 0.3  # 評価に用いるデータの割合
verbose = 2             # 学習進捗の表示モード
decay = 0.005            # 減衰率
optimizer = Adam(lr=0.0001)      # 最適化アルゴリズム
# callbacks = [make_tensorboard(set_dir_name='log')]  # コールバック


# ニューラルネットワークの設計
# 浸透サブネット
input_img = Input(shape=(maindt_size+subdt_size,))
x = input_img
for i in range(layers_percnet):
    if i == layers_percnet - 1:
        x = Dense(percfeature_size)(x)
    else:
        x = Dense(percnet_size)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
feature = x

# 浸透サブネットの設定
percnet = Model(input_img, feature)
percnet.compile(optimizer=optimizer, loss=mean_squared_error)

# 全体のネットワークの設定
x = percnet.output
for i in range(layers_intnet - 1):
    x = Dense(intnet_size)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
x = Dense(output_size)(x)
output = Activation('softmax')(x)

network = Model(percnet.input, output)
network.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])


# MNISTデータの読み込み
(x_train_aux, y_train), (x_test_aux, y_test) = mnist.load_data()
y_train = to_categorical(y_train, output_size)
y_test = to_categorical(y_test, output_size)
# 各データが0~1の値となるように調整
x_train_aux = x_train_aux.astype('float32') / 255
x_test_aux = x_test_aux.astype('float32') / 255
# 28*28ピクセルのデータを784個のデータに平滑化
x_train_aux = x_train_aux.reshape([len(x_train_aux), subdt_size])
x_test_aux = x_test_aux.reshape([len(x_test_aux), subdt_size])
# 主データの作成
x_train_main = shuffle_data(x_train_aux, shuffle_rate)
x_test_main = shuffle_data(x_test_aux, shuffle_rate)
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
# 入力データの表示
n = 10
plt.figure()
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[-i][:subdt_size].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(x_test[-i][subdt_size:].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

# 事前学習
network.summary()
history_list = LossAccHistory()
network.fit(x_train, y_train,
            epochs=epochs_prior,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=(x_val, y_val),
            callbacks=[history_list])

# 浸透特徴の保存
perc_feature = percnet.predict(x_train)
perc_feature_val = percnet.predict(x_val)

# 浸透学習
epoch = 0
loss = 1
non_perc_rate = 1   # 非浸透率
nprate_min = 1e-8   # 非浸透率の閾値
loss_min = 1e-5     # 損失関数の値の閾値
# 補助データに非浸透率を掛けるための配列を作成
non_perc_vec = np.ones(x_train.shape[1])
non_perc_vec[:subdt_size] = 1 - decay
# 学習
percnet.summary()
while non_perc_rate > nprate_min or loss > loss_min:
    non_perc_rate = (1 - decay) ** epoch
    print('Non Percolation Rate =', non_perc_rate)
    percnet.fit(x_train, perc_feature,
                initial_epoch=epochs_prior+epoch, epochs=epochs_prior+epoch+1,
                batch_size=batch_size,
                verbose=verbose,
                validation_data=(x_val, perc_feature_val))
    ev = network.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
    ev_val = network.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
    history_list.losses.append(ev[0])
    history_list.accuracy.append(ev[1])
    history_list.losses_val.append(ev_val[0])
    history_list.accuracy_val.append(ev_val[1])
    loss = percnet.evaluate(x_train, perc_feature, verbose=0)
    x_train *= non_perc_vec
    epoch += 1
    if epoch >= epochs_perc:
        break

# 微調整
if True:    # Trueには微調整する条件を入れる(現状は常に微調整を行う)
    non_perc_vec[:subdt_size] = 0
    x_train *= non_perc_vec
    network.fit(x_train, y_train,
                initial_epoch=epochs_prior+epochs_perc,
                epochs=epochs_prior+epoch+epochs_adj,
                batch_size=batch_size,
                verbose=verbose,
                validation_data=(x_val, y_val),
                callbacks=[history_list])

score_train = network.evaluate(x_train, y_train, batch_size=batch_size)
score_val = network.evaluate(x_val, y_val, batch_size=batch_size)
score_test = network.evaluate(x_test, y_test, batch_size=batch_size)
print('Train - loss:', score_train[0], '- accuracy:', score_train[1])
print('Validation - loss:', score_val[0], '- accuracy:', score_val[1])
print('Test - loss:', score_test[0], '- accuracy:', score_test[1])

plt.figure()
plt.plot(history_list.accuracy)
plt.plot(history_list.accuracy_val)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.0, 1.01)
plt.legend(['Train', 'Test'])
plt.show()

plt.figure()
plt.plot(history_list.losses)
plt.plot(history_list.losses_val)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

