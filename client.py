import socket
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import sys
import tensorflow as tf
from tensorflow import keras
from utility import preprocess
from mlp import initialiseMLP
import pandas as pd
# Initialize a local model
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


if len(sys.argv) != 2:
    print("Usage: python client.py <client_id>")
    sys.exit(1)

client_n = int(sys.argv[1])
print(client_n)


# Assign each client with training images from two classes
if client_n == 0:
    idx0 = y_train == client_n
    idx1 = y_train == 9
else:
    idx0 = y_train == client_n
    idx1 = y_train == client_n - 1
X_train_client = np.concatenate((X_train[idx0][:len(X_train[idx0])//2], X_train[idx1][len(X_train[idx1])//2:]))
y_train_client = np.concatenate((y_train[idx0][:len(y_train[idx0])//2], y_train[idx1][len(y_train[idx1])//2:]))

print('Shape of client {} data: '.format(client_n), X_train_client.shape)
print("Client labels", y_train_client.flatten())
unique_labels = np.unique(y_train_client)

# Print the unique class labels
print("Unique class labels:", unique_labels)

tf.random.set_seed(1)
keras.backend.clear_session()

# Initialize the CNN model

# Train the client model
def trainClient(client_model, batch_size=10, epochs=1):
    print("Training client ", str(client_n))
    client_model.fit(X_train_client, y_train_client, epochs=10, batch_size=32, validation_split=0.1)

# Helper to send data
def send_with_length(sock, data):
    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    sock.sendall(data_length.to_bytes(4, 'big'))  # Send length as 4 bytes
    sock.sendall(serialized_data)  # Send actual data

# Helper to receive data
def recv_with_length(sock):
    data_length = int.from_bytes(sock.recv(4), 'big')  # Read length first
    data = b""
    while len(data) < data_length:
        packet = sock.recv(data_length - len(data))
        if not packet:
            raise ConnectionError("Incomplete data received")
        data += packet
    return pickle.loads(data)

# Main client logic
local_model = initialiseMLP(input_dim, lr=0.1)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 8080))

for round_num in range(100):  # Number of communication rounds
    print(f"--- Round {round_num + 1} ---")
    print("labels for training: ", y_train_client)
    trainClient(local_model)
    
    local_weights = local_model.get_weights()
    send_with_length(client, local_weights)
    
    global_weights = recv_with_length(client)

    # Update the local model
    local_model.set_weights(global_weights)
    print("Updated local model with global weights.")

client.close()
print("Client disconnected.")


'''
import mlsocket
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import sys
import tensorflow as tf
from tensorflow import keras
# Initialize a local model
def create_model():
    model = Sequential([
        Dense(32, input_shape=(10,), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#local_model = create_model()

# Simulated local training data
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))


if len(sys.argv) != 2:
    print("Usage: python client.py <client_id>")
    sys.exit(1)

client_n = int(sys.argv[1])
print(client_n)

#importing and preprocessing data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

X_train = X_train/255
X_train = X_train.astype(float)
y_train = y_train.astype(int)

# Assign each client with training images from two classes
if client_n == 0:
    idx0 = y_train == client_n
    idx1 = y_train == 9
else:
    idx0 = y_train == client_n
    idx1 = y_train == client_n - 1

X_train_client = np.concatenate((X_train[idx0][:len(X_train[idx0])//2], X_train[idx1][len(X_train[idx1])//2:]))
y_train_client = np.concatenate((y_train[idx0][:len(y_train[idx0])//2], y_train[idx1][len(y_train[idx1])//2:]))

X_train_client_cnn = np.expand_dims(X_train_client, -1)

print('Shape of client {} data: '.format(client_n), X_train_client.shape)
print("Client labels", y_train_client)

tf.random.set_seed(1)
keras.backend.clear_session()


# 
def initialiseCNN(lr=0.1):
    model = Sequential([

        keras.Input(shape=(28, 28, 1)),

        keras.layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        keras.layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=lr),
                  metrics=['accuracy'])

    return model

def trainClient(client_model, batch_size = 10, epochs = 1):

    print("Training client ", str(client_n))
    client_model.fit(X_train_client_cnn, y_train_client, shuffle = True,
                                     batch_size=batch_size,epochs=epochs)


# Helper to send data
def send_with_length(sock, data):
    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    sock.sendall(data_length.to_bytes(4, 'big'))  # Send length as 4 bytes
    sock.sendall(serialized_data)  # Send actual data

# Helper to receive data
def recv_with_length(sock):
    data_length = int.from_bytes(sock.recv(4), 'big')  # Read length first
    data = b""
    while len(data) < data_length:
        packet = sock.recv(data_length - len(data))
        if not packet:
            raise ConnectionError("Incomplete data received")
        data += packet
    return pickle.loads(data)



local_model = initialiseCNN(lr=0.125)     

client = mlsocket.MLSocket()
client.connect(('127.0.0.1', 8080)) 

for round_num in range(20):  # Number of communication rounds
    print(f"--- Round {round_num + 1} ---")
    print("labels for training: ", y_train_client)
    trainClient(local_model)
    
    local_weights = local_model.get_weights()
    send_with_length(client, local_weights)

    
    global_weights = recv_with_length(client)

    # Update the local model
    local_model.set_weights(global_weights)
    print("Updated local model with global weights.")

client.close()
print("Client disconnected.")
'''