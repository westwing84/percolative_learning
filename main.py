# 浸透学習の実装

from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

maindt_size = 784       # 主データのサイズ
subdt_size = 784        # 補助データのサイズ
percnet_size = 100      # 浸透サブネットの各層の素子数
percfeature_size = 100  # 浸透特徴の個数
intnet_size = 100       # 統合サブネットの各層の素子数
output_size = 10        # 出力データのサイズ
epochs_prior = 100      # 事前学習のエポック数
epochs_perc = 1000      # 浸透学習のエポック数
epochs_adj = 200        # 微調整のエポック数
batch_size = 256        # バッチサイズ
validation_split = 0.0  # 評価に用いるデータの割合
decay = 0.05            # 減衰率
optimizer = Adam()      # 最適化アルゴリズム

# ニューラルネットワークの設計
# 浸透サブネット
input_img = Input(shape=(maindt_size+subdt_size,))
x = BatchNormalization()(input_img)
x = Dense(percnet_size, activation='relu')(x)
x = BatchNormalization()(x)
feature = Dense(percfeature_size, activation='relu')(x)
# 統合サブネット
x = BatchNormalization()(feature)
x = Dense(intnet_size, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(intnet_size, activation='relu')(x)
output = Dense(output_size, activation='softmax')(x)

# 全体のネットワークの設定
network = Model(input_img, output)
# 浸透サブネットの設定
percnet = Model(input_img, feature)

# コンパイル
network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
percnet.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

# MNISTデータの読み込み
(x_train_aux, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train, output_size)
y_test = to_categorical(y_test, output_size)
# 各データが0~1の値となるように調整
x_train_aux = x_train_aux.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# 28*28ピクセルのデータを784個のデータに平滑化
x_train_aux = x_train_aux.reshape((len(x_train_aux), -1))
x_test = x_test.reshape((len(x_test), -1))
# 主データの作成
noise_factor = 0.5
x_train_main = x_train_aux + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_aux.shape)
x_train_main = np.clip(x_train_main, 0., 1.)
# 主データと補助データを結合
x_train = np.concatenate([x_train_aux, x_train_main], axis=1)

'''
# 入力データの表示
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_train_aux[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(x_train_main[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

# 事前学習
network.summary()
network.fit(x_train, y_train,
            epochs=epochs_prior,
            batch_size=batch_size,
            validation_split=validation_split)
# 浸透特徴の保存
perc_feature = percnet.predict(x_train)
print(perc_feature.shape)

# 浸透学習
epoch = 0
loss = 1
non_perc_rate = 1   # 非浸透率
nprate_min = 1e-5   # 非浸透率の閾値
loss_min = 1e-3     # 損失関数の値の閾値
while non_perc_rate > nprate_min or loss > loss_min:
    non_perc_rate = (1 - decay) ** epoch
    x_train = np.concatenate([non_perc_rate * x_train_aux, x_train_main], axis=1)
    percnet.fit(x_train, perc_feature,
                initial_epoch=epoch, epochs=epoch+1,
                batch_size=batch_size,
                validation_split=validation_split)
    loss = percnet.evaluate(x_train, perc_feature, verbose=0)[0]
    epoch += 1
    if epoch >= epochs_perc:
        break

# 微調整
if True:    # Trueには微調整する条件を入れる(現状は常に微調整を行う)
    non_perc_rate = 0
    x_train = np.concatenate([non_perc_rate * x_train_aux, x_train_main], axis=1)
    network.fit(x_train, y_train,
                epochs=epochs_adj,
                batch_size=batch_size,
                validation_split=validation_split)
