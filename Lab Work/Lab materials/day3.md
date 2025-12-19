# Measuring the Quality of Fit

## MSE

$$
MSE = \frac1n\sum\left[y_i - \hat f(x_i)\right]^2
$$

Usually, this MSE is computed using training data, and so should more accurately be referred to as the training MSE.

If we have a test data set, we can select a model that gives the smallest test MSE. If we don't we use information criteria such as AIC, BIC, and others. This cannot be used for all models.

### The Bias-Variance Trade-Off

When a model makes predictions, its total expected error can be decomposed into three parts:

1\. Bias

-   Error from incorrect assumptions in the model.

-   High bias → model is too simple (underfitting).

2\. Variance

-   Error from sensitivity to small fluctuations in the training data.

-   High variance → model is too complex (overfitting).

3\. Irreducible Error (Noise)

-   Error from randomness in the data.

| Problem | Cause | Symptoms | Fixes |
|------------------|------------------|------------------|------------------|
| **Underfitting** | High bias | Low train & test accuracy | More complex model |
| **Overfitting** | High variance | High train, low test accuracy | Regularization, more data |

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

# Model Selection

## Best Subset Selection:

This method tries **all possible combinations of predictors** and chooses the subset that gives the best model according to

### With No testing data

1.  R-square / adj R-square

    $$
    AdjustedR^2 = 1 − \frac{RSS/(n − d − 1)}{TSS/(n − 1)}
    $$

2.  C_p / AIC

    $$
    \begin{align}
    C_p &= \frac1n \left(RSS +2d\hat \sigma^2\right)\\
    AIC &= −2\log(L) + 2d
    \end{align}
    $$

3.  BIC

    $$
    BIC =\frac1n\left(RSS + \log(n)dσˆ2\right)
    $$

### With testing data

```{r}
test_predictions11 <- predict(model11, newdata = test)
test_predictions22 <- predict(model22, newdata = test)
test_predictions31 <- predict(model31, newdata = test)

# Calculate the Test MSE
test_actuals <- test$Sales
test_mse11 <- mean((test_actuals - test_predictions11)^2)
test_mse22 <- mean((test_actuals - test_predictions22)^2)
test_mse31 <- mean((test_actuals - test_predictions31)^2)

# Print the Test MSE
print(paste("Test MSE Sales ~ TV:", test_mse11))
print(paste("Test MSE Sales ~ TV + Radio:", test_mse22))
print(paste("Test MSE Sales ~ TV + Radio + Newspaper:", test_mse31))
```