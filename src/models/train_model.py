import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # Pegando a pasta que este arquivo se encontra
    featurePath = os.path.dirname(os.path.abspath(__file__))
    # Pasta raíz do projeto
    rootPath = os.path.dirname(os.path.dirname(featurePath))

    STOCK_PATH = os.path.join(rootPath, "data", "processed", "2015-2018", "PETR.csv")
    SAVE_PATH = os.path.join(rootPath, "models", "PETR")
    os.makedirs(SAVE_PATH, exist_ok=True)

    # the features avalibels are [volume, open, max, min, close]
    # so if you wanna select volume put 0, or max put 2
    FEATURE_PREDICT = 4
    # how many days in the future : e.g: 1 = next day predict, 3 = 3 days in the future predict
    DAYS_IN = 1
    # how many days should be considered to predict the next
    TIME_STEP = 5
    BATCH_SIZE = 8
    EPOCH = 150

    NUMBER_OF_FEATURES = 5
    # Loading the dataset
    # [codneg, volume, open, max, min, close]
    dataset_train = pd.read_csv(STOCK_PATH)

    # Featuring Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(dataset_train.iloc[:, 1:6])

    # Selecting the last 5 coluns
    features = training_set_scaled[:-1, 0:6]
    # Selecting the result from the first day that whants to predict
    answeres = training_set_scaled[(TIME_STEP+(DAYS_IN-1)):, FEATURE_PREDICT]

    x = np.reshape(features[0:TIME_STEP],(-1, NUMBER_OF_FEATURES, TIME_STEP))

    for i in range(TIME_STEP, (features.shape[0]-TIME_STEP)):
        reshaped = np.reshape(features[i:(i+TIME_STEP)],(-1, NUMBER_OF_FEATURES, TIME_STEP))
        x = np.append(x, reshaped,0)

    # taking the answeres 
    y = answeres[:x.shape[0]]

    # Reshaping y
    y = np.reshape(y,(-1,1))

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.callbacks import CSVLogger
    from keras.callbacks import ModelCheckpoint

    regressor = Sequential()

    regressor.add(LSTM(units = 8, return_sequences = True, input_shape = (TIME_STEP,NUMBER_OF_FEATURES )))
    regressor.add(LSTM(units = 16, return_sequences = True))
    regressor.add(LSTM(units = 32, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 64, return_sequences = True))
    regressor.add(Dropout(0.5))

    regressor.add(Dense(units = 1, activation = "sigmoid"))

    regressor.compile(optimizer = 'adam', loss = 'mse', metrics=['mae'])

    csv_logger = CSVLogger(os.path.join(SAVE_PATH, 'your_log_name' + '.log'), append=True)
    save_model = os.path.join(SAVE_PATH, "model-{epoch:02d}-{loss:.3f}-{mae:.4f}.model")

    checkpoint1 = ModelCheckpoint(save_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(save_model, monitor='mae', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint1, checkpoint2, csv_logger]

    regressor.fit(x, y, epochs = EPOCH, batch_size = BATCH_SIZE, callbacks= callbacks_list)

if __name__ == '__main__':
    main()