import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt

def main():
    print("Starting prediction")
    # Pegando a pasta que este arquivo se encontra
    featurePath = os.path.dirname(os.path.abspath(__file__))
    # Pasta raíz do projeto
    rootPath = os.path.dirname(os.path.dirname(featurePath))

    MODEL_PATH = os.path.join(rootPath, "models", "PETR", "model-03-0.002-0.0303.model")
    DATASET_PATH = os.path.join(rootPath, "data", "processed", "2019", "PETR.csv")
    save_dir = os.path.join(rootPath, "reports", "validacao", "PETR")
    os.makedirs(save_dir)
    # the features avalibels are [volume, open, max, min, close]
    # so if you wanna select volume put 0, or max put 2
    FEATURE_PREDICT = 4

    # how many days in the future : e.g: 1 = next day predict, 3 = 3 days in the future predict
    DAYS_IN = 1
    # how many days should be considered to predict the next
    TIME_STEP = 5
    NUMBER_OF_FEATURES = 5
    dataset_train = pd.read_csv(DATASET_PATH)

    # Featuring Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    test_set_scaled = sc.fit_transform(dataset_train.iloc[:, 1:6])

    # Selecting the last 5 coluns
    test = test_set_scaled[:-1, 0:5]
    # Selecting the result from the first day that whants to predict
    result_expected = test_set_scaled[(TIME_STEP+(DAYS_IN-1)):, FEATURE_PREDICT]

    x = np.reshape(test[0:TIME_STEP],(-1,TIME_STEP,NUMBER_OF_FEATURES))

    for i in range(TIME_STEP, (test.shape[0]-TIME_STEP)):
        reshaped = np.reshape(test[i:(i+TIME_STEP)],(-1,TIME_STEP,NUMBER_OF_FEATURES))
        x = np.append(x, reshaped,0)

    # taking the answeres 
    y = result_expected[:x.shape[0]]
    print("Load Model")
    model = load_model(MODEL_PATH)
    
    yhat = model.predict(x, verbose=0)

    #rmse = sqrt(mean_squared_error(y, yhat))

    # ---------- Ploting
    print("Ploting graph")
    plt.plot(y, color = 'blue', label = 'Esperado')
    plt.plot(yhat, color = 'red', label = 'Predito')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "VALIDACAO.png"), format='png')
    print("Done")
if __name__ == "__main__":
    main()