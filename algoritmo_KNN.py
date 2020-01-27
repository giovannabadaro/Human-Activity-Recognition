# Importando Biliotecas Importantes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

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

def classificador():
  classifier = KNeighborsClassifier(n_neighbors = 12)
  classifier.fit(x_train, y_train)
  y_pred = classifier.predict(x_test)
  return y_pred

# Gerando a Matriz de Confusão com os dados de teste
def matrix_confusion(y_test,y_pred):
  cm = confusion_matrix(y_test, y_pred)
  return cm

def metrics (y_test, y_pred):
  ac = accuracy_score(y_test, y_pred)
  rec = recall_score(y_test,y_pred, average= 'weighted')
  prec = precision_score(y_test,y_pred, average='weighted')
  return (ac, rec, prec)


  ##calculo do tempo de execução do classificador
tempo_inicial = time.time()
y_pred = classificador()
print("--- %s segundos ---" % (time.time() - tempo_inicial))

acuracia, recall, precisao= metrics(y_test,y_pred)
stringSaida= "acuracia = {ac} , recall = {rec} , precisao = {prec} ".format (ac = acuracia, rec = recall, prec = precisao)
print (stringSaida)

cm = matrix_confusion(y_test,y_pred)
print(cm)






