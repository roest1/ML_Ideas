# model.py
'''

Modeling thoughts:

* Sequence to day time series deep learning

* need to be able to predict OGValue for each tank for future day. 


Data Quality Concerns:

* If tank status is 0, don't predict next value. Else if status is 1 or nan predict that tank

* If tankOGTentativeQuality = is 0, don't predict the next value. 
* If tankProcessedTrueQUality is 0, don't predict the next value. 

If any one of these is 0, don't predict the next value. 

How can we update the model architecture (or do we?) so that it can still descern between which features (columns) or nodes in the net our tanks 
refer to and when to tell the model not to output predictions for that next day if any of these are 0? I want to refrain from predicting these, because
I believe the bad quality data will tank the model performance. If we can avoid testing accuracy on these tanks for days they show up 0 for one of these attributes,
I think we can have a good test accuracy. 

If the model says its not going to predict a tank for the next day, we need to think of a placeholder so we can identify when the model chose not to predict the days. 

'''
import tensorflow as tf
from preprocess import Config, TANKS
import numpy as np
from dataclasses import dataclass
import time

def masked_loss(y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    mask = tf.cast(mask, tf.float64)  # Convert mask to float
    loss = tf.reduce_sum(mask * tf.square(y_true - y_pred)) / \
        (tf.reduce_sum(mask) + tf.keras.backend.epsilon())
    return loss

@dataclass
class ModelParams:
    model:tf.keras.Sequential
    optimizer:tf.keras.optimizers.Optimizer
    loss_fn:callable
    train_mse:tf.keras.metrics.Mean
    val_mse:tf.keras.metrics.Mean
    batch_size:int
    epochs:int
    train_data:tf.data.Dataset
    val_data:tf.data.Dataset
    test_data:tf.data.Dataset


def CNN_Model(C: Config) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(
            shape=(C.num_input_days, len(C.feature_cols)),
            sparse=False,
            dtype=tf.float64,
        ),
        tf.keras.layers.Conv1D(
            filters=64,  # dimension of output
            kernel_size=15,  # conv window (days)
            strides=1,
            padding='valid',  # no padding
            data_format='channels_last',  # (batch, days, tanks)
            dilation_rate=1,
            groups=1,
            activation='relu',
            use_bias=True,
            dtype=tf.float64,
        ),
        tf.keras.layers.MaxPool1D(
            pool_size=3,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dtype=tf.float64,
        ),
        tf.keras.layers.Conv1D(
            filters=32,  # dimension of output
            kernel_size=5,  # conv window
            strides=1,
            padding='valid',  # no padding
            data_format='channels_last',  # (batch, days, tanks)
            dilation_rate=1,
            groups=1,
            activation='relu',
            use_bias=True,
            dtype=tf.float64,
        ),
        tf.keras.layers.MaxPool1D(
            pool_size=3,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dtype=tf.float64,
        ),
        tf.keras.layers.Flatten(data_format='channels_last'),
        tf.keras.layers.Dense(
            units=128,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            dtype=tf.float64,
        ),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(
            units=64,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            dtype=tf.float64,
        ),
        tf.keras.layers.Dense(
            units=len(C.target_cols),
            activation='linear',
            dtype=tf.float64,
        ),
    ])

@tf.function
def train_step(M:tf.keras.Sequential, optimizer:tf.keras.optimizers.Optimizer, loss_fn:callable, train_mse, X_batch, y_batch, mask_batch):
    # what are the types of X_batch, y_batch, and mask_batch? 
    with tf.GradientTape() as tape:
        logits = M(X_batch, training=True)
        loss = loss_fn(y_batch, logits, mask_batch)
    grads = tape.gradient(loss, M.trainable_weights)
    optimizer.apply_gradients(zip(grads, M.trainable_weights))
    train_mse.update_state(loss)

@tf.function
def val_step(M:tf.keras.Sequential, loss_fn:callable, val_mse, X_batch, y_batch, mask_batch):
    logits = M(X_batch, training=False)
    loss = loss_fn(y_batch, logits, mask_batch)
    val_mse.update_state(loss)

def train_model(P:ModelParams):
    start_time = time.time()
    train_loss, val_loss = [], []
    for epoch in range(P.epochs):
        
        # TRAINING
        for X_batch, y_batch, mask_batch in P.train_data:
            train_step(P.model, P.optimizer, P.loss_fn, P.train_mse, X_batch, y_batch, mask_batch)


        train_loss.append(P.train_mse.result())
        P.train_mse.reset_states()
        
        # VALIDATION
        for X_batch, y_batch, mask_batch in P.val_data:
            val_step(P.model, P.loss_fn, P.val_mse, X_batch, y_batch, mask_batch)
            
        val_loss.append(P.val_mse.result())
        P.val_mse.reset_states()

        print(f'Epoch {epoch + 1}/{P.epochs} | Train MSE = {train_loss[-1].numpy():.4f} | Val MSE = {val_loss[-1].numpy():.4f}') 

    print(f"Total train time = {time.time() - start_time}s")


def filter_invalid_predictions(predictions: np.ndarray, masks: np.ndarray) -> np.ndarray:
    # Replace invalid predictions with a placeholder (e.g., -1)
    return np.where(masks == 1, predictions, -1)
