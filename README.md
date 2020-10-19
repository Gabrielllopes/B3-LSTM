B3-LSTM
==============================



<img src=/reports/figures/rede.png>

### Este projeto implementa uma Rede Neural Artificial LSTM que prevê os valores de fechamento de ações da B3.  

### Neste repositório são encontrados os scripts necessários para a análise e predição de ações da B3.

Organização do projeto
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Dados intermediários de manipulação.
    │   ├── processed      <- Dados processados e prontos para utilizar no treinamento.
    │   └── raw            <- Exemplo de dados originais obtidos na B3.
    │
    ├── models             <- Modelos treinados.
    │
    ├── references         <- Referências utilizas para criar este trabalho.
    │
    ├── reports            
    │   └── figures        <- Gráficos das ações utilizadas.
    │
    ├── src                <- Códfigo utilizado neste projeto.
    │   │
    │   ├── features       <- Manipula os dados para que eles fiquem no formato de treino.
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts para treinar e predizer os modelos.
    │   │   ├── predict_model.py
    │   │   ├── grid_search.py <- Realiza a busca em grid por hyper parametros
    │   │   └── train_model.py <- Treina um modelo
    │   │
    │   └── visualization 
    └──      └── visualize.py <- Script para vizualizar os dados utilizados para treinamento e test

## Exemplos de Predições

<img src=/reports/figures/bbdc.png>
<img src=/reports/figures/vale.png>

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
