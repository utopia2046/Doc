# Evaluation Metrics

## Common evaluation functions

### Regression evaluation functions

* MSE: Mean Square Error, RMSE, R-squared
$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y-\hat{y})^2$$
$$RMSE = \sqrt{MSE}$$
* MAE: Mean Absolute Error (MAE is more stable when your dataset contains outliers)
$$
MAE(y, \hat y) = \frac{1}{N} \sum_{i=1}^N |\hat y_i - y_i|
$$
* (R)MSPE, MAPE
* (R)MSLE

### Classification evaluation functions
* Accurary (how frequently our class prediction is correct)
$$Accuracy = \frac{1}{N} \sum_{i=1}^{N} [\hat{y_i}=y_i]$$
* LogLoss
$$LogLoss = \frac{1}{N} \sum_{i=1}^{N} y_i\log(\hat{y_i}) + (1-y_i)log(1-\hat{y_i}) \qquad y_i \in \mathbb{R}, \hat{y_i} \in \mathbb{R}$$
* AUC (ROC): Area Under Curve
* Cohen's (Quadratic weighted) Kappa

