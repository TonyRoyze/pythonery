# Measuring the Quality of Fit

## MSE

$$
MSE = \frac1n\sum\left[y_i - \hat f(x_i)\right]^2
$$

Usually, this MSE is computed using training data, and so should more accurately be referred to as the training MSE.

If we have a test data set, we can select a model that gives the smallest test MSE. If we don't we use information criteria such as AIC, BIC, and others. This cannot be used for all models.

### Overfitting & Underfitting

$MSE_{Tr}$ and $MSE_{Te}$ regardless of the data set and the model used. When a model has a small $MSE_{Tr}$and a very large $MSE_{Te}$, then it is said to be overfitting the data. Regardless of whether overfitting has occurred or not, we generally expect $MSE_{Tr} < MSE_{Te}$ .

\$\$ \begin{align}

\end{align} \$\$

### The Bias-Variance Trade-Off

What is the reason for the U-shape observed in MSETe curves? This is due to two competing properties of statistical learning methods. Suppose we estimate f using a training data set, and let (x0, y0) be a test observation drawn from the population. If the true model is Y = f (X) + ϵ (here f (x) = E(Y \|X = x)), then E  y0 − fˆ(x0) 2 = V(fˆ(x0)) + h Bias(fˆ(x0))i2 + V(ϵ), where Bias  fˆ(x0)  = E h fˆ(x0) i − f (x0). Here E  y0 − fˆ(x0) 2 is the expected square loss at X = x0.

## Task

```{python}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(123)
X = np.linspace(0, 10, 50)
y_true = np.sin(X) + 0.1 * X
y_noisy = y_true + np.random.normal(0, 0.5, size=X.shape)  # sd = 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
```

1.  Fit Polynomial Models (Increasing Complexity)

    1.  Fit polynomial regressions with degrees 1, 3, 5, 9, 15 on the training data.

        ```{python}
        degrees = [1, 3, 5, 9, 15]
        train_mse, test_mse = [], [] 
        x_fit = np.linspace(0, 10, 300).reshape(-1, 1)

        for deg in degrees: 
            poly = PolynomialFeatures(degree=deg) 
            X_train_poly = poly.fit_transform(X_train) 
            X_test_poly = poly.transform(X_test) 
            model = LinearRegression() 
            model.fit(X_train_poly, y_train) 
            y_train_pred = model.predict(X_train_poly) 
            y_test_pred = model.predict(X_test_poly) 
            y_fit = model.predict(poly.transform(x_fit)) 
            train_mse.append(mean_squared_error(y_train, y_train_pred))
            test_mse.append(mean_squared_error(y_test, y_test_pred))
        ```

    2.  Compute train MSE and test MSE for each degree.

        ```{python}
        print("Degree | Train MSE | Test MSE")      
        print("-------------------------------")      
        for deg, tr, te in zip(degrees, train_mse, test_mse):          
                print(f"{deg:6d} | {tr:9.4f} | {te:8.4f}")
        ```

2.  Visualization – Model Fits vs True Function Plot the training and test points, the true function (black), and predicted curves for each fitted model.

    ```{python}
    for deg in degrees: 
        plt.plot(x_fit, y_fit, label=f'Degree {deg}')

    plt.scatter(X_train, y_train, color='blue', label='Train data') 
    plt.scatter(X_test, y_test, color='red', label='Test data') 
    plt.plot(X, y_true, color='black', linewidth=2, label='True function')

    plt.title('Polynomial Regression Fits (Various Degrees)') 
    plt.xlabel('X') 
    plt.ylabel('y') 
    plt.legend() 
    plt.show()
    ```

3.  Visualization – Train vs Test Error Plot train MSE and test MSE against polynomial degree on the same chart.

    ```{python}
    plt.figure(figsize=(8, 5))     
    plt.plot(degrees, train_mse, 'o-', label='Train MSE', color='blue')     
    plt.plot(degrees, test_mse, 's--', label='Test MSE', color='red')     
    plt.xlabel('Polynomial Degree')     
    plt.ylabel('Mean Squared Error')     
    plt.title('Training vs Testing Error by Polynomial Degree')     
    plt.legend()     
    plt.show()
    ```