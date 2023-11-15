import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Carregam el conjunt de dades
train = pd.read_csv('./dat/fashion-mnist_train.csv')
test = pd.read_csv('./dat/fashion-mnist_test.csv')

# Separam el conjunt d'entrenament en característiques X_train i etiquetes Y_train
X_train= train.drop(['label'],axis = 1)
y_train = train['label']

# Dimensió del conjunt de característiques
print(X_train.shape)

# Dimensió de les etiquetes
print(y_train.shape)

# Dividir el conjunto de entrenamiento en subconjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Inicializar el clasificador SVM
svm_classifier = SVC()

# Entrenar el modelo SVM
svm_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de validación
y_val_pred = svm_classifier.predict(X_val)

# Evaluar el rendimiento en el conjunto de validación
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Accuracy on validation set: {accuracy}")
print("hey")

# Ahora, puedes aplicar el modelo entrenado al conjunto de prueba
#X_test = test_data  # Ajusta esto según cómo estén estructurados tus datos de prueba
# y_test_pred = svm_classifier.predict(X_test)