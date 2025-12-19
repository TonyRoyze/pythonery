## Ridge Regression, Lasso Regression, and Elastic Net (with Equations + R Code)

### Ridge Regression
**Equation:**  
Minimize  
\[
\text{RSS} + \lambda \sum_{j=1}^{p} \beta_j^2 
= \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \|\beta\|_2^2
\]  
- Uses an **L2 penalty**.  
- Shrinks coefficients but **never to zero**.  
- Helps with multicollinearity and reduces variance.  
- Keeps all predictors in the model.

**R code (glmnet):**
```r
library(glmnet)

X <- model.matrix(y ~ ., data = df)[, -1]
y <- df$y

ridge_model <- glmnet(X, y, alpha = 0)      # alpha=0 → ridge
plot(ridge_model)
cv_ridge <- cv.glmnet(X, y, alpha = 0)
best_lambda <- cv_ridge$lambda.min
ridge_coefs <- coef(cv_ridge, s = "lambda.min")
