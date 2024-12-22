import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import metrics
import tensorflow_datasets as tfds
import numpy as np
from utility import preprocess


##The model
def initialiseMLP(input_dim, lr=0.1):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax'),
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=lr),
                  metrics=['accuracy'])

    return model



models = {
    'mlp':initialiseMLP,
}
'''
data = pd.read_csv("srm.csv")

attack_mapping = {
    'Benign': 0,
    'Exploits': 1,
    'Reconnaissance': 2,
    'DoS': 3,
    'Generic': 4,
    'Shellcode': 5,
    'Backdoor': 6,
    'Fuzzers': 7,
    'Worms': 8,
    'Analysis': 9
}

X_train, X_test, y_train, y_test = preprocess(data,attack_mapping)


input_dim = X_train.shape[1]
mlp_model = initialiseMLP(input_dim, lr=0.01)
'''
#mlp_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)