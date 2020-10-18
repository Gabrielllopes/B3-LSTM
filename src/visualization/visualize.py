"""
@author: gabriel
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    # Pegando a pasta que este arquivo se encontra
    featurePath = os.path.dirname(os.path.abspath(__file__))
    # Pasta raíz do projeto
    rootPath = os.path.dirname(os.path.dirname(featurePath))

    # Gráficos de todas as ações da pasta processed 2019
    # OBS: tammbém é possivel fazer os gráficos de 2015-2018 com este script
    path_processed = os.path.join(rootPath, "data", "processed", "2019")
    for arquivo in os.listdir(path_processed):
        filename = arquivo.split(".")[0]
        print("Criando gráfico de {}".format(arquivo))
        file_path = os.path.join(path_processed, arquivo)
        preco_fechamento(file_path, filename, os.path.join(rootPath, "reports", "figures", filename+".png"))

# descobre a variância de uma ação
def variance(data):
    opening = np.array(data.iloc[:,2])
    close = np.array(data.iloc[:,5])
    
    var = (close/opening) - 1 
    var[np.isnan(var)] = 0
    var_mean = var.mean()
    
    variance = math.sqrt((((var - var_mean)**2).sum()) / (len(var) -1))
    return variance

# Preço de fechamento no tempo
def preco_fechamento(caminho_do_arquivo, nome_arquivo, save_dir):
    arq = pd.read_csv(caminho_do_arquivo)
    
    var = round(variance(arq), 4)
      
    sns.set()
    plt.figure(figsize=(16,8))
    plt.plot(arq.index, arq.PREULT)
    plt.grid()
    plt.xlabel("Dias", fontsize=20)
    plt.ylabel("Preço de fechamento", fontsize=20)
    plt.title(nome_arquivo, fontsize=35)
    plt.figtext(.15, .82, "Variância: {}".format(var), fontsize=15)
    
    plt.savefig(save_dir, format='png')
    
if __name__ == "__main__":
    main()
    