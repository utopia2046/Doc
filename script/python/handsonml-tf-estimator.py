# https://tensorflow.google.cn/guide/migrate/canned_estimators?hl=zh-cn
# pip install tensorflow_decision_forests
import keras
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_decision_forests as tfdf

# prepare training/test data
x_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
x_eval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
x_train['sex'].replace(('male', 'female'), (0, 1), inplace=True)
x_eval['sex'].replace(('male', 'female'), (0, 1), inplace=True)

x_train['alone'].replace(('n', 'y'), (0, 1), inplace=True)
x_eval['alone'].replace(('n', 'y'), (0, 1), inplace=True)

x_train['class'].replace(('First', 'Second', 'Third'), (1, 2, 3), inplace=True)
x_eval['class'].replace(('First', 'Second', 'Third'), (1, 2, 3), inplace=True)

x_train.drop(['embark_town', 'deck'], axis=1, inplace=True)
x_eval.drop(['embark_town', 'deck'], axis=1, inplace=True)

y_train = x_train.pop('survived')
y_eval = x_eval.pop('survived')

# Data setup for TensorFlow 1 with `tf.estimator`
def _input_fn():
  return tf1.data.Dataset.from_tensor_slices((dict(x_train), y_train)).batch(32)

def _eval_input_fn():
  return tf1.data.Dataset.from_tensor_slices((dict(x_eval), y_eval)).batch(32)

FEATURE_NAMES = [
    'age', 'fare', 'sex', 'n_siblings_spouses', 'parch', 'class', 'alone'
]

feature_columns = []
for fn in FEATURE_NAMES:
  feat_col = tf1.feature_column.numeric_column(fn, dtype=tf.float32)
  feature_columns.append(feat_col)

# create optimizer that works for both v1 and v2
def create_sample_optimizer(tf_version):
  if tf_version == 'tf1':
    optimizer = lambda: tf.keras.optimizers.legacy.Ftrl(
        l1_regularization_strength=0.001,
        learning_rate=tf1.train.exponential_decay(
            learning_rate=0.1,
            global_step=tf1.train.get_global_step(),
            decay_steps=10000,
            decay_rate=0.9))
  elif tf_version == 'tf2':
    optimizer = tf.keras.optimizers.legacy.Ftrl(
        l1_regularization_strength=0.001,
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.1, decay_steps=10000, decay_rate=0.9))
  return optimizer

# TF v1 DNNEstimator
dnn_estimator = tf.estimator.DNNEstimator(
    head=tf.estimator.BinaryClassHead(),
    feature_columns=feature_columns,
    hidden_units=[128],
    activation_fn=tf.nn.relu,
    optimizer=create_sample_optimizer('tf1'))
dnn_estimator.train(input_fn=_input_fn, steps=100)
dnn_estimator.evaluate(input_fn=_eval_input_fn, steps=10)

# TF v2 Customized
dnn_model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(1)])
# layer options like tf.keras.layers.RNN, tf.keras.layers.LSTM, tf.keras.layers.GRU

dnn_model.compile(loss='mse', optimizer=create_sample_optimizer('tf2'), metrics=['accuracy'])

dnn_model.fit(x_train, y_train, epochs=10)
dnn_model.evaluate(x_eval, y_eval, return_dict=True)
