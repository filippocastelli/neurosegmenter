from typing import Callable
import tensorflow as tf
from tensorflow.keras import backend as K


def weighted_cross_entropy_loss(pos_weight: float) -> Callable:
    if pos_weight is None:
        raise ValueError("need to define a pos_weight parameter in training config to use weighted_cross_entropy_loss")

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        p_hat = tf.math.log(y_pred / (1 - y_pred))
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=p_hat, labels=y_true, pos_weight=pos_weight)
        return tf.reduce_mean(loss)

    return loss


def weighted_categorical_crossentropy_loss(weights) -> Callable:
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        import pdb
        pdb.set_trace()
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss