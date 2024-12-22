import socket
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from sklearn.metrics import accuracy_score
from mlp import models, initialiseMLP
from utility import preprocess
import pandas as pd
# Load data and test set
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
global_model = initialiseMLP(input_dim, lr=0.1)


global_weights = global_model.get_weights()

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

def modelAccuracy(model):
    prob = model.predict(X_test)
    pred = np.argmax(prob, axis=-1)

    print("Model can predict classes: ", np.unique(pred))
    print("Model accuracy: {}".format(accuracy_score(y_test, pred)),
          "\n-------------------------------------------------------------")
    return accuracy_score(y_test, pred)

# Server setup
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 8080))
server.listen()

clients = []
num_clients = 9  # Number of clients
rounds = 100       # Number of communication rounds

print("Waiting for clients to connect...")

# Accept connections from clients
for _ in range(num_clients):
    conn, addr = server.accept()
    clients.append(conn)
    print(f"Connected to client: {addr}")

print("All clients connected. Starting Federated Learning...")
global_accuracy = []
# Start Federated Learning
for round_num in range(rounds):
    print(f"--- Round {round_num + 1} ---")
    aggregated_weights = [np.zeros_like(weight) for weight in global_weights]
    valid_clients = 0

    # Collect updates from clients
    for conn in clients:
        try:
            client_weights = recv_with_length(conn)
            aggregated_weights = [aggregated_weights[i] + client_weights[i] for i in range(len(global_weights))]
            valid_clients += 1
        except Exception as e:
            print(f"Error receiving data from a client: {e}. Skipping.")

    if valid_clients == 0:
        print("No valid updates received. Skipping this round.")
        continue

    # Average the weights
    global_weights = [weight / valid_clients for weight in aggregated_weights]

    # Update the global model
    global_model.set_weights(global_weights)
    model_accuracy = modelAccuracy(global_model)
    global_accuracy.append(model_accuracy)
    print("Model accuracy history: ", global_accuracy)
    # Broadcast updated global model to clients
    for conn in clients:
        try:
            send_with_length(conn, global_weights)
        except Exception as e:
            print(f"Error sending data to a client: {e}")

print("Federated Learning complete. Closing connections.")
print("Saving globa model")
global_model.save("global_model_nf.h5")
for conn in clients:
    conn.close()


'''
import mlsocket
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

# Initialize a global model
def create_model():
    model = Sequential([
        Dense(32, input_shape=(10,), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

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


global_model = initialiseCNN()
global_weights = global_model.get_weights()


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


# Server setup
server = mlsocket.MLSocket()
server.bind(('0.0.0.0', 8080))
server.listen()

clients = []
num_clients = 3  # Number of clients
rounds = 20       # Number of communication rounds

print("Waiting for clients to connect...")

# Accept connections from clients
for _ in range(num_clients):
    conn, addr = server.accept()
    clients.append(conn)
    print(f"Connected to client: {addr}")

print("All clients connected. Starting Federated Learning...")

# Start Federated Learning
for round_num in range(rounds):
    print(f"--- Round {round_num + 1} ---")
    aggregated_weights = [np.zeros_like(weight) for weight in global_weights]
    valid_clients = 0

    # Collect updates from clients
    for conn in clients:
        try:
            #print(f"Waiting for data from client {client_id}...")
            client_weights = recv_with_length(conn)
            #print(f"Received weights from client {client_id}.")
            aggregated_weights = [aggregated_weights[i] + client_weights[i] for i in range(len(global_weights))]
            valid_clients += 1
        except Exception as e:
            print(f"Error receiving data from a client: {e}. Skipping.")

    if valid_clients == 0:
        print("No valid updates received. Skipping this round.")
        continue

    # Average the weights
    global_weights = [weight / valid_clients for weight in aggregated_weights]

    # Update the global model
    global_model.set_weights(global_weights)

    # Broadcast updated global model to clients
    for conn in clients:
        try:
            send_with_length(conn, global_weights)
            #print(f"Sent updated weights to client {client_id}.")
        except Exception as e:
            print(f"Error sending data to a client: {e}")

print("Federated Learning complete. Closing connections.")
for conn in clients:
    conn.close()

'''    
