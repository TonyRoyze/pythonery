import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(np):
    def sigmoid(x):
      return 1/(1+np.exp(-x))

    return (sigmoid,)


@app.cell
def _(np, sigmoid):
    lr = 0.001
    n_iters = 1000
    X = np.array([[1,2,3], [4,5,6]])
    y = np.array([0,1,0])
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    for _ in range(n_iters):
      linear_pred = np.dot(X, weights) + bias
      predictions = sigmoid(linear_pred)
      dw = (1/n_samples) * np.dot(X.T, (predictions-y))
      db = (1/n_samples) * np.sum(predictions-y)

      weights = weights - lr * dw
      bias = bias - lr * db


    linear_pred = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_pred)
    class_pred = [0 if y<=0.5 else 1 for y in y_pred]
    class_pred
    return


@app.cell
def _(np, sigmoid):
    class Logistic:
      def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

      def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
          linear_pred = np.dot(X, self.weights) + self.bias
          predictions = sigmoid(linear_pred)
          dw = (1/n_samples) * np.dot(X.T, (predictions-y))
          db = (1/n_samples) * np.sum(predictions-y)

          self.weights = self.weights - self.lr * dw
          self.bias = self.bias - self.lr * db

      def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred

    return


if __name__ == "__main__":
    app.run()
