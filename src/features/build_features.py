"""
@author: gabriel
"""

import pandas as pd
import os
import csv

# Este main é um exemplo de como utilizar as funções para construção de um dataset de treinamento
def main():
    # Pegando a pasta que este arquivo se encontra
    featurePath = os.path.dirname(os.path.abspath(__file__))
    # Pasta raíz do projeto
    rootPath = os.path.dirname(os.path.dirname(featurePath))

    # Pasta em que os dados não processados se encontram
    # Para este exemplo eu coloquei dados históricos retirados direto do site da B3
    dados_historicos = os.path.join(rootPath, "data", "raw", "COTAHIST_A2019.TXT")
    # Os dados intermediarios serão salvos na pasta interim
    dados_intermediarios_path = os.path.join(rootPath, "data", "interim")

    # primeiro os dados são convertidos para csv
    converte_cotacoes(arquivoLer=dados_historicos,
    nome_de_salvamento=os.path.join(dados_intermediarios_path, "dados_intermediarios.csv"))
    # em seguida seleciona apenas os dados de lote padrão, e organiza eles por nome
    organiza_acoes(caminho_do_arquivo=os.path.join(dados_intermediarios_path, "dados_intermediarios.csv"),
    nome_de_salvamento=os.path.join(dados_intermediarios_path, "dados_agrupados.csv"))
    # separando apenas os dados utilizados para o treinamento dos modelos
    junta_acoes(caminho_do_arquivo=os.path.join(dados_intermediarios_path, "dados_agrupados.csv"),
    nome_de_salvamento=os.path.join(dados_intermediarios_path, "dados_final.csv"))
    # Por fim separa as ações por nome
    separando_por_nome(caminho_do_arquivo=os.path.join(dados_intermediarios_path, "dados_final.csv"),
    nome_salvamento=os.path.join(rootPath, "data", "processed"))


# Convertendo cotações históricas da bolsa
# Atencao ! Se esta funcao for executada mais de uma vez replicara os dados.
def converte_cotacoes(arquivoLer , nome_de_salvamento):
    print(" Convertendo cotações de txt para csv.\n Arquivo de leitura:({})\n Arquivo de escrita:({})".format(arquivoLer, nome_de_salvamento))

    # setando arquivo de leitura
    arquivoLeitura = open(arquivoLer,'r')
    
    # escrevendo o nome das coluns primeiro
    colunas = [ 'Data','CODBI','CODNEG','TPMERC','NORMES'
                   ,'ESPECI','PRAZOT','MODREF','PREABE','PREMAX',
                   'PREMIN','PREMED','PREULT','PREOFC','PREOFV','TOTNEG',
                   'QUATOT','VOLTOT','PREEXE','INDOPC','DATVEN','FATCOT',
                   'PTOEXE','CODISI','DISMES' ]
    
    with open(nome_de_salvamento, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(colunas)
        
    csvFile.close()  
    
    for linha in arquivoLeitura:
        dicionario = [
                        linha[2:10],
                        linha[10:12],
                        linha[12:24],
                        linha[24:27],
                        linha[27:39],
                        linha[39:49],
                        linha[49:52],
                        linha[52:56],
                        linha[56:69],
                        linha[69:82],
                        linha[82:95],
                        linha[95:108],
                        linha[108:121],
                        linha[121:134],
                        linha[134:147],
                        linha[147:152],
                        linha[152:170],
                        linha[170:188],
                        linha[188:201],
                        linha[201:202],
                        linha[202:210],
                        linha[210:217],
                        linha[217:230],
                        linha[230:242],
                        linha[242:244]
                      ]
        with open(nome_de_salvamento, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(dicionario)
        
        csvFile.close()       
    arquivoLeitura.close()


def organiza_acoes(caminho_do_arquivo, nome_de_salvamento):
    print(" Organizando e agrupando ações.\n Arquivo de leitura:({})\n Arquivo de escrita:({})".format(caminho_do_arquivo, nome_de_salvamento))
    arq = pd.read_csv(caminho_do_arquivo,dtype={'PRAZOT': str})
    
    # organiza as acoes por nome
    arq = arq.sort_values(by=['CODNEG','Data'],axis = 0)
    arq = arq.reset_index(drop=True)
        
    # converte alguns numeros de string para int
    arq.PREABE = arq.PREABE.astype('int64') 
    arq.PREMAX = arq.PREMAX.astype('int64') 
    arq.PREMIN = arq.PREMIN.astype('int64') 
    arq.PREMED = arq.PREMED.astype('int64') 
    arq.PREULT = arq.PREULT.astype('int64') 
    arq.PREOFC = arq.PREOFC.astype('int64') 
    arq.PREOFV = arq.PREOFV.astype('int64')
    arq.TOTNEG = arq.TOTNEG.astype('int64')
    arq.QUATOT = arq.QUATOT.astype('int64') 
    arq.VOLTOT = arq.VOLTOT.astype('int64') 
    arq.PREEXE = arq.PREEXE.astype('int64') 
    
    # Seleciona apenas lote padrão codbi 02 
    arq = arq[(arq.CODBI == 2)]
    
    # resetando index
    arq = arq.reset_index(drop=True)
    
    arq.to_csv(nome_de_salvamento, index=False)

# Esta função permite agrupar as ações para que sejam treinadas
# dias_a_juntar = quantos dias vão compor uma entrada da rede
# dia_de_prever = quantos dias a frente quer prever o valor
def junta_acoes(caminho_do_arquivo, nome_de_salvamento, dias_a_juntar = 1, dia_de_prever = 1):
    print(" Cria dataset de treino.\n Arquivo de leitura:({})\n Arquivo de escrita:({})".format(caminho_do_arquivo, nome_de_salvamento))
    arq = pd.read_csv(caminho_do_arquivo)

    aux = dias_a_juntar
    #coluna auxiliar de salvamentode salvamento
    ms = []
    #matris de salvamente
    matSv = []

    i = 0
    while(i < (arq.shape[0]-1)):
        if(i+dias_a_juntar < arq.shape[0]-1):
            if(arq.CODNEG[i] == arq.CODNEG[i+dias_a_juntar]):
                ms.append(arq.CODNEG[i])

                while(aux > 0):
                    ms.append(arq.VOLTOT[i])
                    ms.append(arq.PREABE[i])
                    ms.append(arq.PREMAX[i])
                    ms.append(arq.PREMIN[i])
                    ms.append(arq.PREULT[i])
                    
                    aux = aux - 1
                    i = i + 1

                i = i-1
                aux = dias_a_juntar
                matSv.append(ms)
                ms = []
        i = i+1

    arquivoFinal = pd.DataFrame(matSv)

    arquivoFinal.columns

    dia_futuro = (5 * dia_de_prever) + 1

    arquivoFinal.rename(columns ={0:"CODNEG",
                                  dia_futuro:"DIA_PREV",
                                  arquivoFinal.shape[1]-1:"PREULT"
                                  },
                        inplace=True)
    
    arquivoFinal.to_csv(nome_de_salvamento, index=False)

def separando_por_nome( caminho_do_arquivo,
                        nome_salvamento, 
                        lista_acoes = ["ITUB4       ","VALE3       ","PETR4       ","BBDC4       ",
                                       "B3SA3       ","PETR3       ","ABEV3       ","BBAS3       ",
                                       "ITSA4       ","JBSS3       ","LREN3       "]):
    print(" Separa por nome.\n Arquivo de leitura:({})\n Arquivo de escrita:({})".format(caminho_do_arquivo, nome_salvamento))
    arq = pd.read_csv(caminho_do_arquivo)

    for i in lista_acoes:
        arq[arq.CODNEG == i].to_csv(os.path.join(nome_salvamento, str(i[:4])+".csv"), index=False)

if __name__ == '__main__':
    main()