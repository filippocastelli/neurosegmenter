from typing import Callable
import tensorflow as tf


def weighted_cross_entropy_loss(pos_weight: float) -> Callable:
    if pos_weight is None:
        raise ValueError("need to define a pos_weight parameter in training config to use weighted_cross_entropy_loss")

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        p_hat = tf.math.log(y_pred / (1 - y_pred))
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=p_hat, labels=y_true, pos_weight=pos_weight)
        return tf.reduce_mean(loss)

    return loss
