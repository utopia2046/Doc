# Model Optimization

## Tune Hyperparameters

1. Select the most influential parameters;
2. Understand, how exactly they influence the training;
3. Manually change and examine (or automatically, hyperopt, etc.)

### Random Forrest and other Tree-based Models Hyperparameters

Parameters that increase accuracy:

* XGBoost: max_depth, subsample, colsample_bytree, colsample_bylevel, eta, num_round;
* LightGBM: max_depth, numleaves, bagging_fraction, leaning_rate, num_iterations;
* sklearn.RandomForest/ExtraTrees: max_depth, max_features;

Parameters that reduce overfitting:

* XGBoost: min_child_weight, lambda, alpha;
* LightGBM: min_data_in_leaf, lambda_l1, lambda_l2;
* sklearn.RandomForest/ExtraTrees: min_samples_leaf;

Parameters that don't affect model too much:

* XGBoost, random seed;
* LightGBM: \*_seed;
* sklearn.RandomForest/ExtraTrees: criterion, random_state, njobs;

### What framework to use

* Keras ✓, Lasagne
* TensorFlow
* MxNet
* PyTorch ✓
* sklearn's NLP

### Tune Neural Nets

Parameters that increase accuracy:

* Number of neurons per layer
* Number of layers
* Optimizer: Adam/Adadelta/Adagrad/...
* Batch size

Parameters that reduce overfitting:

* Optimizer: SGD + Momentum
* L2/L1 for weights
* Dropout/Dropconnect
* Static dropconnect

### Tune Linear Models

Parameters that increase accuracy:

* Number of neurons per layer
* Number of layers
* Optimizer: Adam/Adadelta/Adagrad/...
* Batch size

Parameters that reduce overfitting:

* Regularizion parameters (C, alpha, lambda, ...)

### Tips

* Don't spend too much time tuning hyperparameters (Only if you don't have any more ideas or you have spare computational resources)
* Be patient (It can take thousands of rounds for GBDT or neural nets to fit)
* Average everything
  * Over random seed
  * Or over small deviations from optimal parameters (e.g. average max_depth=4,5,6 for an optimal 5)

### References

* <http://fastml.com/optimizing-hyperparams-with-hyperopt/>
* <https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/>
* <http://scikit-learn.org/stable/modules/grid_search.html>
