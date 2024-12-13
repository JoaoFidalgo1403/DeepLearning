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

############################################################################################################################
# On the making
class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.W1 = .1 * np.random.normal(0.1, 0.1, (hidden_size, n_features)) 
        self.b1 = .1 * np.zeros((hidden_size))                             
        self.W2 = .1 * np.random.normal(0.1, 0.1, (n_classes, hidden_size)) 
        self.b2 = .1 * np.zeros((n_classes))                            

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def forward(self, x, weights, biases, debug):
        num_layers = len(weights)
        g = self.relu
        hiddens = []
        # compute hidden layers
        for i in range(num_layers):
                h = x if i == 0 else hiddens[i-1]
                z = weights[i].dot(h) + biases[i]
                if i < num_layers - 1:  # Assuming the output layer has no activation.
                    hiddens.append(g(z))
        #compute output
        output = z
        return output, hiddens

# PROBLEM PROBABLY IS HERE        
    def compute_loss(self, output, y):
        # compute loss
        epsilon=1e-6
        output_shifted = output - np.max(output)  # SHIFT VALUES SO EXP DOESNT OVERFLOW
        probs = np.exp(output_shifted) / (np.sum(np.exp(output_shifted)) + epsilon) # ADD EPSILON SO LOG DOESNT BREAK
        loss = -y.dot(np.log(probs + epsilon)) 

        return loss #loss is returning negative values which is not possible! 
        
    def backward(self, x, y, output, hiddens, weights):
        num_layers = len(weights)
        g = self.relu
        z = output

# PROBLEM PROBABLY IS HERE        
        epsilon=1e-6
        output_shifted = output - np.max(output)  # SHIFT VALUES SO EXP DOESNT OVERFLOW
        probs = np.exp(output_shifted) / (np.sum(np.exp(output_shifted)) + epsilon) # ADD EPSILON SO LOG DOESNT BREAK

        grad_z = probs - y

        grad_weights = []
        grad_biases = []
        
        # Backpropagate gradient computations 
        for i in range(num_layers-1, -1, -1):
            
            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)

            grad_h = weights[i].T.dot(grad_z)
            grad_z = grad_h * self.relu_derivative(h)

        # Making gradient vectors have the correct order
        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def predict(self, inputs, weights, biases):
        predicted_labels = []
        for x in inputs:
            # Compute forward pass and get the class with the highest probability
            output, _ = self.forward(x, weights, biases, False)
            y_hat = np.argmax(output)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X, [self.W1, self.W2], [self.b1, self.b2])
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        num_layers = len([self.W1, self.W2])
        total_loss = 0

        # For each observation and target
        for x, y_true in zip(X, y):
            # Compute forward pass for a single sample

            output, hiddens = self.forward(x, [self.W1, self.W2], [self.b1, self.b2], True)
            
            # Compute the loss for a single sample
            loss = self.compute_loss(output, y_true)
            total_loss += loss

            # Compute backpropagation for a single sample
            grad_weights, grad_biases = self.backward(x, y_true, output, hiddens, [self.W1, self.W2])

            # Update weights and biases individually for each layer
            self.W1 -= learning_rate * grad_weights[0]
            self.b1 -= learning_rate * grad_biases[0]
            self.W2 -= learning_rate * grad_weights[1]
            self.b2 -= learning_rate * grad_biases[1]

            return total_loss
                
############################################################################################################################


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    #plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    #plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    #plt.show()


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
    parser.add_argument('-data_path', type=str, default='../../intel_landscapes.v2.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, init_train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(init_train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)

    if opt.model == 'mlp':
        one_hot_train = np.zeros((init_train_y.shape[0], n_classes))
        for i in range(init_train_y.shape[0]):
            one_hot_train[i, init_train_y[i]] = 1
        train_y = one_hot_train
    else:
        train_y = init_train_y  # No need to one-hot encode for non-mlp models
        
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, init_train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        init_train_y = init_train_y[train_order]
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
        
        train_accs.append(model.evaluate(train_X, init_train_y))
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
