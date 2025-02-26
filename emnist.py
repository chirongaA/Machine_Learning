
import matplotlib
matplotlib.use('TkAgg')# Switch to the TkAgg backend for interactive plots 
matplotlib.use('Agg')# Use Agg backend for rendering plots to files

import struct
import numpy as np
import matplotlib.pyplot as plt

# Function to read IDX files 
def read_idx(file_path):
    with open(file_path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic == 0x801:	# Labels
            return np.frombuffer(f.read(), dtype=np.uint8)
        elif magic == 0x803:	# Images
            rows, cols = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows * cols)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Weight initialization
def initialize_weights(input_size, hidden_size, output_size):
    weights = {
        'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size),
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size),
        'b2': np.zeros((1, output_size)),
}
    return weights

# Forward and backward pass
def forward_pass(X, weights):
    z1 = np.dot(X, weights['W1']) + weights['b1'] 
    a1 = relu(z1)
    z2 = np.dot(a1, weights['W2']) + weights['b2'] 
    return z1, a1, z2

def backward_pass(X, y, z1, a1, z2, weights, gradients, lr, momentum, velocity, reg_lambda):
    m = X.shape[0]
    dz2 = z2 - y
    gradients['W2'] = (np.dot(a1.T, dz2) / m) + reg_lambda * weights['W2'] 
    gradients['b2'] = np.sum(dz2, axis=0, keepdims=True) / m
    dz1 = np.dot(dz2, weights['W2'].T) * relu_derivative(z1)
    gradients['W1'] = (np.dot(X.T, dz1) / m) + reg_lambda * weights['W1'] 
    gradients['b1'] = np.sum(dz1, axis=0, keepdims=True) / m
    for key in weights:
        velocity[key] = momentum * velocity[key] - lr * gradients[key] 
        weights[key] += velocity[key]

# Training loop with mini-batch gradient descent
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
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            z1, a1, z2 = forward_pass(X_batch, weights)
            backward_pass(X_batch, y_batch, z1, a1, z2, weights, gradients, lr, momentum, velocity, reg_lambda)

            # Calculate training loss
            _, _, z2_full = forward_pass(X_train, weights) 
            loss = np.mean((z2_full - y_train) ** 2) 
            loss_history.append(loss)

            # Early stopping logic 
            if loss < min_loss:
                min_loss = loss 
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Adjust learning rate 
            lr *= decay_rate
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}") 
    return weights, loss_history
    
# Predict function
def predict(X, weights):
    _, _, z2 = forward_pass(X, weights) 
    return np.argmax(z2, axis=1)

# Accuracy calculation
def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels) * 100

# Main program
if   __name__== "__main__":
    dataset_prefix = input("Enter dataset prefix (e.g., emnist-digits, emnist-balanced): ")
    print(f"Using {dataset_prefix} dataset.")
 
    # Paths to dataset files
    train_images_path = f"emnist/{dataset_prefix}-train-images-idx3-ubyte" 
    train_labels_path = f"emnist/{dataset_prefix}-train-labels-idx1-ubyte" 
    test_images_path = f"emnist/{dataset_prefix}-test-images-idx3-ubyte"
    test_labels_path = f"emnist/{dataset_prefix}-test-labels-idx1-ubyte"

    # Load dataset
    train_images = read_idx(train_images_path) 
    train_labels = read_idx(train_labels_path)
    test_images = read_idx(test_images_path) 
    test_labels = read_idx(test_labels_path)

    # Normalize images
    train_images = train_images / 255.0 
    test_images = test_images / 255.0

    # One-hot encode labels
    num_classes = len(np.unique(train_labels))
    train_labels_encoded = np.zeros((train_labels.size, num_classes)) 
    train_labels_encoded[np.arange(train_labels.size), train_labels] = 1

    # Training parameters 
    input_size = 784
    hidden_size = 256 
    output_size = num_classes
    epochs = 100
    batch_size = 64
    lr = 0.001
    decay_rate = 0.95
    momentum = 0.9
    patience = 5
    reg_lambda = 0.001

    # Train the model
    print("Training the model...")
    trained_weights, loss_history = train(train_images, train_labels_encoded, input_size, hidden_size, output_size, epochs, batch_size, lr, decay_rate, momentum, patience, reg_lambda)
    
    # Plot training loss
    plt.plot(range(len(loss_history)), loss_history) 
    plt.title('Training Loss')
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.savefig('training_loss.png') 
    plt.show()

    # Evaluate model on test data
    print("Evaluating the model...")
    predictions = predict(test_images, trained_weights) 
    accuracy = calculate_accuracy(predictions, test_labels) 
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Visualize predictions 
    for i in range(5):
        plt.imshow(test_images[i].reshape(28, 28), cmap='gray') 
        plt.title(f"Predicted: {predictions[i]}, Actual: {test_labels[i]}")
        plt.savefig(f'prediction_{i}.png') 
        plt.show()
