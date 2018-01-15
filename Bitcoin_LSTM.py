# -*- coding: utf-8 -*-
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

class Prediction :

    def __init__(self):
        self.timesteps = 50
        self.data_dim = 7
        self.hidden_neurons = [1,2]
        self.data_file = "./pubnub/log/test_json.json"

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
        model.add(LSTM(self.hidden_neurons[0],
                            batch_input_shape=(None, self.timesteps, self.data_dim),
                            return_sequences=True))
        model.add(LSTM(self.hidden_neurons[1],return_sequences=False))
        model.add(Dense(self.data_dim))
        model.add(Activation("linear"))
        model.compile(loss="mape", optimizer="adam")
        return model

    def train(self, X_train, y_train) :
        model = self.create_model()
        # 学習
        model.fit(X_train, y_train, batch_size=512, nb_epoch=1)
        return model

if __name__ == "__main__":

    prediction = Prediction()

    # データ準備
    data = pd.read_json(prediction.data_file)
    data = data.drop(["tick_id","product_code", "volume", "volume_by_product"], axis=1)

    # 終値のデータを標準化
    data = data.sort_values(by='timestamp')
    data = data.reset_index(drop=True)
    data = data.drop("timestamp", axis=1)
    data = pd.DataFrame(preprocessing.scale(data))

    # 1割をテストデータへ
    split_pos = int(round(len(data) * 0.9))
    x_train, y_train = prediction.load_data(data.iloc[0:split_pos], prediction.timesteps)
    x_test,  y_test  = prediction.load_data(data.iloc[split_pos:], prediction.timesteps)

    model = prediction.train(x_train, y_train)

    predicted = model.predict(x_test)
    result = pd.DataFrame(predicted[:,2])
    result.columns = ['predict']
    result['actual'] = y_test[:,2]
    result.plot()
    plt.show()