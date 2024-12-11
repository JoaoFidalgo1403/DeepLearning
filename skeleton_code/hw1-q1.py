#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

# Done - Check if it is correct!
class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_pred = np.argmax(np.dot(self.W, x_i))

        if y_pred != y_i: 
            self.W[y_i] += x_i  
            self.W[y_pred] -= x_i  

# Done - Check if it is correct!
class LogisticRegression(LinearModel):

    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """

        # Compute label scores (num_classes x 1)
        label_scores = np.expand_dims(self.W.dot(x_i), axis=1)  # Shape: (num_classes, 1)

        # Compute label probabilities using softmax (num_classes x 1)
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        
        # Create one-hot encoding for y_i (num_classes x 1)
        y_one_hot = np.zeros((np.size(self.W, 0), 1))  # Shape: (num_classes, 1)
        y_one_hot[y_i] = 1

        # Compute gradient and update weights
        # Gradient: (num_classes x num_features)
        gradient = (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis=1).T) - l2_penalty * self.W

        # Update weights
        self.W += learning_rate * gradient

# On the making
class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.W1 = np.random.normal(0.1, 0.1, (n_features, hidden_size)) * np.sqrt(2. / n_features) # Input to hidden layer weights
        self.b1 = np.zeros((1, hidden_size))                             # Bias for hidden layer
        self.W2 = np.random.normal(0.1, 0.1, (hidden_size, n_classes)) * np.sqrt(2. / hidden_size)  # Hidden to output layer weights
        self.b2 = np.zeros((1, n_classes))                              # Bias for output layer

    def to_one_hot(self, y, n_classes):
        return np.eye(n_classes)[y]  # y is an array of shape (batch_size,)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def forward(self, x, weights, biases):
        num_layers = len(weights)
        g = self.relu
        hiddens = []
        # compute hidden layers
        for i in range(num_layers):
                h = x if i == 0 else hiddens[i-1]
                z = np.dot(h, weights[i]) + biases[i] # Needs checking
                if i < num_layers - 1:  # Assuming the output layer has no activation.
                    hiddens.append(g(z))
        #compute output
        output = z
       
        return output, hiddens

    def compute_loss(self, output, y):
        # One-hot encode y to match the shape of probs
        y_one_hot = self.to_one_hot(y, 6)  # y_one_hot has shape (batch_size, n_classes)

        # Numerical stability trick: subtract the max value from output before applying exp
        output_stable = output - np.max(output, axis=1, keepdims=True)
        probs = np.exp(output_stable) / np.sum(np.exp(output_stable), axis=1, keepdims=True)

        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1)

        # Cross-entropy loss: compute element-wise log and multiply with one-hot labels
        loss = -np.mean(np.sum(y_one_hot * np.log(probs), axis=1))

        return loss  # Scalar loss

    def backward(self, x, y, output, hiddens, weights):
        num_layers = len(weights)
        g = self.relu
        z = output

        # Numerical stability trick: subtract the max value from output before applying exp
        output_stable = output - np.max(output, axis=1, keepdims=True)
        probs = np.exp(output_stable) / np.sum(np.exp(output_stable), axis=1, keepdims=True)

        y_one_hot = self.to_one_hot(y, 6)  # y_one_hot has shape (12630, 6)
        grad_z = probs - y_one_hot

        print(f"Grad_z shape: {grad_z.shape}")  # Should be (batch_size, n_classes) 
        
        grad_weights = []
        grad_biases = []
        
        # Backpropagate gradient computations 
        for i in range(num_layers-1, -1, -1):
            
            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]
            grad_weights.append(np.dot(h.T, grad_z)) # Needs Checking
            grad_biases.append(np.sum(grad_z, axis=0)) # Needs Checking 
            
            grad_weights = [np.clip(g, -1, 1) for g in grad_weights]
            grad_biases = [np.clip(g, -1, 1) for g in grad_biases]

            grad_h = np.dot(grad_z, weights[i].T) * g(h) > 0  # Backpropagate ReLU gradient
            grad_z = grad_h * self.relu_derivative(h)

        # Making gradient vectors have the correct order
        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        # Forward pass
        output, _ = self.forward(X, [self.W1, self.W2], [self.b1, self.b2])
        return np.argmax(output, axis=1)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Train the MLP for one epoch using stochastic gradient descent.
        Arguments:
        X -- Input data (n_examples x n_features)
        y -- One-hot encoded labels (n_examples x n_classes)
        learning_rate -- Learning rate for gradient updates
        Returns:
        Average loss for the epoch
        """
       # Perform one epoch of training
        output, hiddens = self.forward(X, [self.W1, self.W2], [self.b1, self.b2])
        loss = self.compute_loss(output, y)

        grad_weights, grad_biases = self.backward(X, y, output, hiddens, [self.W1, self.W2])

        # Update weights and biases using gradient descent
        self.W1 -= learning_rate * grad_weights[0]
        self.b1 -= learning_rate * grad_biases[0]
        self.W2 -= learning_rate * grad_weights[1]
        self.b2 -= learning_rate * grad_biases[1]

        return loss


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='../intel_landscapes.v2.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
