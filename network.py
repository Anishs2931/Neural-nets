import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import os
import json

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid_activation(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        n_test = len(test_data)
        n = len(training_data)
        print("Training starting...")

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.update_parameters(batch, eta)
                self.batch_counter += 1
            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def update_parameters(self, batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid_activation(z)
            activations.append(activation)

        delta = (activations[-1] - y) * sigmoid_activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_activation_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_w, nabla_b
    
    def save(self,file):
        data={
            "sizes":self.sizes,
            "weights":[w.tolist() for w in self.weights],
            "biases":[b.tolist() for b in self.biases]
        }

        with open(file,'w') as f:
             json.dump(data,f)

        print('Network saved to',file)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def predict(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.show()

        img = cv2.resize(img, (28, 28))
        if np.mean(img) > 128:
            img = 255 - img
        img = img / 255.0

        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.title("Preprocessed Image (28x28)")
        plt.show()
        
        img_vector = img.reshape(784, 1)
        
        output = self.feedforward(img_vector)
        prediction = np.argmax(output)
        
        probabilities = output.flatten()
        plt.figure(figsize=(10, 4))
        plt.bar(range(10), probabilities)
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title(f'Prediction: {prediction}')
        plt.xticks(range(10))
        plt.show()
        
        print(f"The network predicts this image is a {prediction}")
        return prediction
    
    def load(self,file):
        with open(file,'r') as f:
            data=json.load(f)
    
        self.sizes=[data["sizes"]]
        self.weights=[np.array(w) for w in data["weights"]]
        self.biases=[np.array(b) for b in data["biases"]]
    
        print("Network loaded")

            
def sigmoid_activation(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_activation_prime(z):
    return sigmoid_activation(z) * (1 - sigmoid_activation(z))