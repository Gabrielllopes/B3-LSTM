import numpy as np
import pandas as pd
import uuid
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from math import sqrt 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from numba import cuda
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# script utilizado para achar os hyperparametros por grid search
def main():
    # Pegando a pasta que este arquivo se encontra
    featurePath = os.path.dirname(os.path.abspath(__file__))
    # Pasta raíz do projeto
    rootPath = os.path.dirname(os.path.dirname(featurePath))
    # aonde vai salvar os modelos gerados pelo grid
    save_path = os.path.join(rootPath, "models", "grid")
    # verifica se a pasta exite caso não, cria uma
    os.makedirs(save_path, exist_ok=True)
    # acão utilizada para treinamento dos modelos em grid
    stock_path  = os.path.join(rootPath, "data", "processed", "2015-2018", "PETR.csv")
    validation_data_path = os.path.join(rootPath, "data", "processed", "2019", "PETR.csv")
    # as features disponíveis são [volume, open, max, min, close]
    # so if you wanna select volume put 0, or max put 2
    # 'volume' : 0,
    # 'open'   : 1,
    # 'max'    : 2,
    # 'min'    : 3,
    # 'close'  : 4
    # selecionar o valor que quer prever
    feature = 'close'
    # quantos dias no futuro vai tentar prever
    days = 1
    # quantos dias vai utilizar como entrada
    time_step = 5

    number_features = 5
    batchSize = 16
    epoch = 100

    # definir os hyperparamteros a serem procurados
    test_layer_list = [[8, 16, 32, 32], [16, 16, 32, 32], [50, 50, 50, 50], [32, 32, 32, 32],
                    [8, 16, 32], [16, 32, 32], [16, 32, 64], [8, 8, 16],
                    [8, 16, 32, 32, 64], [8, 16, 16, 32, 32], [8, 8, 16, 32,32], [8, 16, 16, 32, 64]]
    test_dropout_list = [[0, 0, 0.2, 0.2], [0, 0.2, 0.2, 0.2], [0, 0.2, 0.2, 0.5], [0, 0, 0.2, 0.5],
                        [0,0,0.2], [0,0.2,0.2], [0,0,0.5], [0,0.2,0.5],
                        [0,0,0,0,0], [0,0,0.2,0.2,0.2], [0,0,0,0,0.5], [0,0,0.2,0.3,0.5], [0,0,0,0.3,0.5]]
    test_act_func = ["relu","tanh"]

    searchGrid(save_path, stock_path, feature, days, time_step, number_features, batchSize, validation_data_path, test_layer_list, test_dropout_list, test_act_func, epoch)

def train_model(save_path, stock_path, feature, days,
                time_step, number_features, batchSize,
                layer_list, dropout_list, act_func, epoch):

    # check if the that is consistent
    if len(layer_list) != len(dropout_list):
        raise Exception('The size of layesr is diferent from dropout')


    # The number of layers and the amount of neurons in each layer its going to be defined as a list
    # the size of the list will be the number of layers
    # the number of neurons will be the list members

    STOCK_PATH = stock_path
    SAVE_PATH = save_path

    # the features avalibels are [volume, open, max, min, close]
    # so if you wanna select volume put 0, or max put 2
    feature_dic = {
        'volume' : 0,
        'open'   : 1,
        'max'    : 2,
        'min'    : 3,
        'close'  : 4
        }
    FEATURE_PREDICT = feature_dic.get(feature)

    # how many days in the future : e.g: 1 = next day predict, 3 = 3 days in the future predict
    DAYS_IN = days
    # how many days should be considered to predict the next
    TIME_STEP = time_step
    NUMBER_OF_FEATURES = number_features
    BATCH_SIZE = batchSize

    # Loading the dataset
    # [CODNEG, volume, open, max, min, close]
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

    regressor = Sequential()
    regressor.add(LSTM(units = layer_list[0],
                       return_sequences = True,
                       input_shape = (TIME_STEP,NUMBER_OF_FEATURES )))
    if dropout_list[0] != 0:
        regressor.add(Dropout(dropout_list[0]))

    for i in range(1,(len(layer_list) - 2)):
        regressor.add(LSTM(units = layer_list[i], return_sequences = True))
        if dropout_list[0] != 0:
            regressor.add(Dropout(dropout_list[i]))

    regressor.add(LSTM(units = layer_list[i+1]))
    if dropout_list[0] != 0:
        regressor.add(Dropout(dropout_list[i+1]))

    regressor.add(Dense(units = 1,  activation = act_func))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


    csv_logger = CSVLogger(os.path.join(SAVE_PATH, 'your-0-log_name' + '.log'), append=True)

    save_model = os.path.join(SAVE_PATH, "model-{epoch:02d}-{loss:.3f}.model")

    checkpoint1 = ModelCheckpoint(save_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint1, csv_logger]

    regressor.fit(x, y, epochs = epoch, batch_size = BATCH_SIZE, callbacks= callbacks_list)
    K.clear_session()
    cuda.select_device(0)
    cuda.close()

def validation(model_path, stock_path,  feature, days,
                time_step, number_features ):
    MODEL_PATH = model_path
    DATASET_PATH = stock_path

    # the features avalibels are [volume, open, max, min, close]
    # so if you wanna select volume put 0, or max put 2
    feature_dic = {
    'volume' : 0,
    'open'   : 1,
    'max'    : 2,
    'min'    : 3,
    'close'  : 4
    }
    FEATURE_PREDICT = feature_dic.get(feature)

    # how many days in the future : e.g: 1 = next day predict, 3 = 3 days in the future predict
    DAYS_IN = days
    # how many days should be considered to predict the next
    TIME_STEP = time_step
    NUMBER_OF_FEATURES = number_features
    dataset_test = pd.read_csv(DATASET_PATH)

    # Featuring Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    test_set_scaled = sc.fit_transform(dataset_test.iloc[:, 1:6])

    # Selecting the last 5 coluns
    test = test_set_scaled[:-1,:5]
    # Selecting the result
    result_expected = test_set_scaled[(TIME_STEP+(DAYS_IN-1)):, FEATURE_PREDICT]


    x = np.reshape(test[0:TIME_STEP],(-1,TIME_STEP,NUMBER_OF_FEATURES))

    for i in range(TIME_STEP, (test.shape[0]-TIME_STEP)):
        reshaped = np.reshape(test[i:(i+TIME_STEP)],(-1,TIME_STEP,NUMBER_OF_FEATURES))
        x = np.append(x, reshaped,0)

    y = result_expected[:x.shape[0]]

    model = load_model(MODEL_PATH)

    yhat = model.predict(x, verbose=0)

    rmse = sqrt(mean_squared_error(y, yhat))

    return rmse

def searchGrid(save_path, stock_path, feature, days, time_step, number_features, batchSize, validation_data_path, test_layer_list, test_dropout_list, test_act_func, epoch): 
    # [id, layer config, dropout config, act_func, rmse]
    list_data = []

    for layer_list in test_layer_list:
        for dropout_list in test_dropout_list:
            for act_func in test_act_func:

                if len(layer_list) != len(dropout_list):
                    continue
                else:

                    id_of = str(uuid.uuid1())

                    save_path_aux = os.path.join(save_path, id_of)
                    os.makedirs(save_path_aux, exist_ok=True)

                    train_model(save_path_aux, stock_path, feature, days,
                                    time_step, number_features, batchSize,
                                    layer_list, dropout_list, act_func, epoch)

                    # find the best model
                    all_models = os.listdir(save_path_aux)

                    the_big = 0

                    for i in range(len(all_models)):
                        if int(all_models[i].split('-')[1]) > int(all_models[the_big].split('-')[1]) :
                            the_big = i

                    model_path = os.path.join(save_path_aux,all_models[the_big])

                    mean_squair = validation(model_path, validation_data_path,  feature, days,
                                    time_step, number_features )

                    list_data.append([[id_of],[layer_list],[dropout_list],[act_func],[mean_squair]])
                    print(mean_squair)


    df = pd.DataFrame(data= list_data, columns = ["id", "Network Configuration", "Dropout Configuration", "Activation Function", "RMSE"])
    df.to_csv(os.path.join(save_path, "gird.csv"))

if __name__ == '__main__':
    main()