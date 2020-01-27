# Importando Biliotecas Importantes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

##chamada do modelo
from sklearn.naive_bayes import GaussianNB


nb = GaussianNB()  
nb.fit(x_train, y_train)   
y_predict = nb.predict(x_test)  

##metricas
print("Precisão NB: {:.2f}".format(nb.score(x_test, y_test)))
cm = confusion_matrix(y_test, y_predict)
ac = accuracy_score(y_test, y_predict)
rec = recall_score(y_test,y_predict, average= 'weighted')
prec = precision_score(y_test,y_predict, average='weighted')

##imprimindo as métricas
print (cm)
print("Acurácia:")
print(ac)
print("Recall")
print(rec)
print("Precisão")
print(prec)
