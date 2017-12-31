# -*- coding: utf-8 -*-
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

class Prediction :

    def __init__(self):
        self.length_of_sequences = 10
        self.in_out_neurons = 1
        self.hidden_neurons = 300
        self.data_file = "./pubnub/log/bitflyer_FX_BTC_JPY.json"


    def load_data(self, data, n_prev=10):
        X, Y = [], []
        for i in range(len(data) - n_prev):
            X.append(data.iloc[i:(i+n_prev)].as_matrix())
            Y.append(data.iloc[i+n_prev].as_matrix())
        retX = numpy.array(X)
        retY = numpy.array(Y)
        return retX, retY


    def create_model(self) :
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, \
                            batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), \
                            return_sequences=False))
        model.add(Dense(self.in_out_neurons))
        model.add(Activation("linear"))
        model.compile(loss="mape", optimizer="adam")
        return model


    def train(self, X_train, y_train) :
        model = self.create_model()
        # 学習
        model.fit(X_train, y_train, batch_size=10, nb_epoch=100)
        return model


if __name__ == "__main__":

    prediction = Prediction()

    # データ準備
    data = pd.read_json(prediction.data_file)
    # 終値のデータを標準化
    data['best_bid'] = preprocessing.scale(data['best_bid'])
    data = data.sort_values(by='timestamp')
    data = data.reset_index(drop=True)
    data = data.loc[:, ['timestamp','best_bid']]

    # 2割をテストデータへ
    split_pos = int(len(data) * 0.8)
    x_train, y_train = prediction.load_data(data[['best_bid']].iloc[0:split_pos], prediction.length_of_sequences)
    x_test,  y_test  = prediction.load_data(data[['best_bid']].iloc[split_pos:], prediction.length_of_sequences)

    model = prediction.train(x_train, y_train)

    predicted = model.predict(x_test)
    result = pd.DataFrame(predicted)
    result.columns = ['predict']
    result['actual'] = y_test
    result.plot()
    plt.show()