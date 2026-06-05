import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classification
    """)
    return


@app.cell
def _():
    import numpy as np
    from collections import Counter
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    def euclidean_distance(x1, x2):
      return np.sqrt(np.sum((x1 - x2) ** 2))

    return (
        Counter,
        ListedColormap,
        datasets,
        euclidean_distance,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _(ListedColormap, datasets, plt, train_test_split):
    cmap = ListedColormap( ['#FF0000', '#00FF00', '#0000FF'])
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    plt. figure()
    plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
    plt.show()
    return X_test, X_train, y_test, y_train


@app.cell
def _(Counter, X_test, X_train, euclidean_distance, np, y_test, y_train):
    k = 3

    def predict(x):
      distances = [euclidean_distance(x, x_train) for x_train in X_train]

      k_indices = np.argsort(distances)[:k]
      k_nearest_labels = [y_train[i] for i in k_indices]

      most_common = Counter(k_nearest_labels).most_common()
      return(most_common[0][0])

    predictions = [predict(x) for x in X_test]
    print(predictions)
    print(y_test)
    return (predictions,)


@app.cell
def _(np, predictions, y_test):
    _acc = np.sum(predictions == y_test) / len(y_test)
    print(_acc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Class
    """)
    return


@app.cell
def _(Counter, euclidean_distance, np):
    class KNN:
      def __init__(self, k=3):
        self.k = k
  
      def fit(self, X, y):
        self.X_train = X
        self.y_train = y

      def predict(self, X):
        return [self._predict(x) for x in X]

      def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

    return (KNN,)


@app.cell
def _(KNN, X_test, X_train, np, y_test, y_train):
    clf = KNN(k=5)
    clf.fit(X_train, y_train)
    predictions_1 = clf.predict(X_test)
    _acc = np.sum(np.array(predictions_1) == y_test) / len(y_test)
    print(_acc)
    return


if __name__ == "__main__":
    app.run()
