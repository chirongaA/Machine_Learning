import struct
import numpy as np
 
 
# Function to read IDX files 
def read_idx(file_path):
    with open(file_path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic == 0x801:	# Labels
            return np.frombuffer(f.read(), dtype=np.uint8)
        elif magic == 0x803:	# Images
            rows, cols = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows * cols)

# Activation functions and derivatives 
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))	# Stability for large values
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Weight initialization
def initialize_weights(input_size, hidden_size, output_size): 
    print("Initializing weights with He Initialization for ReLU.") 
    weights = {
        'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size),
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size),
        
        'b2': np.zeros((1, output_size)),
    }
    return weights

# Forward pass
def forward_pass(X, weights):
    z1 = np.dot(X, weights['W1']) + weights['b1']
    a1 = relu(z1)
    z2 = np.dot(a1, weights['W2']) + weights['b2']
    a2 = softmax(z2)	# Apply softmax to get probabilities 
    return z1, a1, z2, a2

# Backward pass
def backward_pass(X, y, z1, a1, z2, a2, weights, gradients, lr, momentum, velocity, reg_lambda):
    m = X.shape[0]
    dz2 = a2 - y	# Using softmax derivative
    gradients['W2'] = (np.dot(a1.T, dz2) / m) + reg_lambda * weights['W2'] 
    gradients['b2'] = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.dot(dz2, weights['W2'].T) * relu_derivative(z1)
    gradients['W1'] = (np.dot(X.T, dz1) / m) + reg_lambda * weights['W1'] 
    gradients['b1'] = np.sum(dz1, axis=0, keepdims=True) / m

    # Update weights with momentum 
    for key in weights:
        velocity[key] = momentum * velocity[key] - lr * gradients[key] 
        weights[key] += velocity[key]

# Mini-batch gradient descent
def create_batches(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]

# Train the network
def train(X_train, y_train, input_size, hidden_size, output_size, epochs, batch_size, lr, decay_rate, momentum, patience, reg_lambda):
    weights = initialize_weights(input_size, hidden_size, output_size) 
    gradients = {key: np.zeros_like(value) for key, value in
                 weights.items()}
    
    velocity = {key: np.zeros_like(value) for key, value in
                weights.items()}
    min_loss = float('inf')
    patience_counter = 0 
    loss_history = []

    for epoch in range(epochs):
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            z1, a1, z2, a2 = forward_pass(X_batch, weights) 
            backward_pass(X_batch, y_batch, z1, a1, z2, a2, weights, gradients, lr, momentum, velocity, reg_lambda)

        # Compute loss (cross-entropy loss)
        _, _, _, a2_full = forward_pass(X_train, weights)
        loss = -np.mean(np.sum(y_train * np.log(a2_full + 1e-7), axis=1)) # Cross-entropy loss
        loss_history.append(loss)

        # Early stopping
        if loss < min_loss: 
            min_loss = loss 
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Decay learning rate 
        lr *= decay_rate
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}") 

    return weights, loss_history

# Predict function
def predict(X, weights):
    _, _, _, a2 = forward_pass(X, weights)
    return np.argmax(a2, axis=1)

# Accuracy calculation
def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels) * 100

# Main program
if   __name__ == "__main__":
    print("Loading and preprocessing the KMNIST dataset.")

    # Paths to the KMNIST files
    train_images_path = 'train-images-idx3-ubyte' 
    train_labels_path = 'train-labels-idx1-ubyte' 
    test_images_path = 't10k-images-idx3-ubyte' 
    test_labels_path = 't10k-labels-idx1-ubyte'

    # Load KMNIST dataset
    train_images = read_idx(train_images_path) 
    train_labels = read_idx(train_labels_path) 
    test_images = read_idx(test_images_path) 
    test_labels = read_idx(test_labels_path)

    # Normalize images to zero mean and unit variance
    print("Normalizing images to zero mean and unit variance.") 
    mean = np.mean(train_images, axis=0)
    std = np.std(train_images, axis=0) 
    train_images = (train_images - mean) / std 
    test_images = (test_images - mean) / std

    # One-hot encode labels
    print("One-hot encoding the labels.") 
    num_classes = 10
    train_labels_encoded = np.zeros((train_labels.size, num_classes))
    train_labels_encoded[np.arange(train_labels.size), train_labels] = 1

    # Train the network
    print("Starting training with advanced techniques.") 
    input_size = 784	# KMNIST images are 28x28 
    hidden_size = 256
    output_size = 10	# 10 classes for KMNIST
    epochs = 100
    batch_size = 64
    lr = 0.001
    decay_rate = 0.95
    momentum = 0.9
    patience = 5
    reg_lambda = 0.001

    trained_weights, loss_history = train(train_images, train_labels_encoded, input_size, hidden_size, output_size, epochs, batch_size, lr, decay_rate, momentum, patience, reg_lambda)

    # Print loss history instead of plotting
    print("Training Loss History (every 10 epochs):")
    for epoch, loss in enumerate(loss_history):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluate on test data
    print("Evaluating on the test dataset.") 
    predictions = predict(test_images, trained_weights)
    accuracy = calculate_accuracy(predictions, test_labels) 
    print(f"Final Test Accuracy: {accuracy:.2f}%")

    # Print sample predictions
    print("Sample predictions (Predicted vs Actual):") 
    for i in range(5):
        print(f"Predicted: {predictions[i]}, Actual: {test_labels[i]}")
