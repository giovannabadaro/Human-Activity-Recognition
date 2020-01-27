# Importando Biliotecas Importantes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn import svm

# Importando funções das bibliotecas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Importando a base de dados
x_train_file = open('/home/giovanna/Documentos/IA/tp02/machine-learning-activities-humans/X_train.txt', 'r')
y_train_file = open('/home/giovanna/Documentos/IA/tp02/machine-learning-activities-humans/y_train.txt', 'r')

x_test_file = open('/home/giovanna/Documentos/IA/tp02/machine-learning-activities-humans/X_test.txt', 'r')
y_test_file = open('/home/giovanna/Documentos/IA/tp02/machine-learning-activities-humans/y_test.txt', 'r')

# Criando listas vazias
x_train = []
y_train = []
x_test = []
y_test = []

# Tabela de mapeamento para classes
labels = {1:'WALKING', 2:'WALKING UPSTAIRS', 3:'WALKING DOWNSTAIRS',
          4:'SITTING', 5:'STANDING', 6:'LAYING'}

# percorrendo as tabelas
for x in x_train_file:
    x_train.append([float(ts) for ts in x.split()])
    
for y in y_train_file:
    y_train.append(int(y.rstrip('\n')))
    
for x in x_test_file:
    x_test.append([float(ts) for ts in x.split()])
    
for y in y_test_file:
    y_test.append(int(y.rstrip('\n')))

# convertendo em numpy para obter maior eficiência
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


def classificadorSVM():
  # criação de um objeto classificador
  model = svm.SVC(kernel='linear', C=1, gamma=1) 
  model.fit(x_train, y_train)
  #variável recebe os valores preditos para os dados de teste
  y_pred= model.predict(x_test)
  return y_pred

#Metricas de desempenho do modelo
acuracia, recall, precisao= metrics(y_test,y_pred)
stringSaida= "acuracia = {ac} , recall = {rec} , precisao = {prec} ".format (ac = acuracia, rec = recall, prec = precisao)
print (stringSaida)
print("\n")

cm = matrix_confusion(y_test,y_pred)
print(cm)
print("\n")

df_confusion = pd.crosstab(y_test, y_pred)
plot_confusion_matrix(df_confusion)
