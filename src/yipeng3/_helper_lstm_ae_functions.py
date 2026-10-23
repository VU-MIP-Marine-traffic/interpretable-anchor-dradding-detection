import tensorflow as tf
from tensorflow.keras import layers

def quantile_loss(tau: float):

    def quantile_loss_function(y_true, y_pred):
        # From the note of the paper, quantile loss can be expresssed as:
        # max( tau * e, -(1-tau) * e ), where e = y_true - y_pred

        less_part = tau * (y_true - y_pred)
        more_part = (1 - tau) * (y_pred - y_true)

        return tf.maximum( less_part, more_part )

    return quantile_loss_function

def lstm_ae(
    tau: float,
    latent_dim: int,
    n_out_features: int,
    n_out_timesteps: int,
    drop_frac: float = 0.1,
    loss_func = quantile_loss,
    *args, **kwargs
):

    model = tf.keras.Sequential([
        layers.LSTM(latent_dim, return_sequences=False),
        layers.Dropout( drop_frac ),

        layers.RepeatVector( n_out_timesteps ),
        layers.LSTM(n_out_features, return_sequences=True),
    ])

    model.compile(optimizer='adam', loss=loss_func(tau))

    return model

def lstm_ae_flat(
    tau: float,
    latent_dim: int,
    n_out_features: int,
    n_out_timesteps: int,
    drop_frac: float = 0.1,
    loss_func = quantile_loss,
    *args, **kwargs
):

    model = tf.keras.Sequential([
        layers.LSTM(latent_dim, return_sequences=False),
        layers.Dropout( drop_frac ),

        layers.RepeatVector( n_out_timesteps ),
        layers.LSTM(n_out_features, return_sequences=True),
        layers.Flatten()
    ])

    model.compile(optimizer='adam', loss=loss_func(tau))

    return model


def lstm_ae_hp(
    tau: float,
    hp,
    n_out_features: int,
    n_out_timesteps: int,
    loss_func = quantile_loss,
    *args, **kwargs
):

    latent_dim = hp.Choice("latent_dim", [5, 10])
    drop_frac = hp.Choice("drop_frac", [0.05, 0.1, 0.25])
    l1 = hp.Choice("l1", [0.01, 0.1, 1.0])

    model = tf.keras.Sequential([
        layers.Dropout( drop_frac ),
        layers.LSTM(
            latent_dim,
            return_sequences=False,
            bias_regularizer=tf.keras.regularizers.L1(l1=l1),
            kernel_regularizer=tf.keras.regularizers.L1(l1=l1),
            recurrent_regularizer=tf.keras.regularizers.L1(l1=l1),
        ),
        layers.Dropout( drop_frac ),
        layers.RepeatVector( n_out_timesteps ),
        layers.LSTM(
            n_out_features,
            return_sequences=True,
            bias_regularizer=tf.keras.regularizers.L1(l1=l1),
            kernel_regularizer=tf.keras.regularizers.L1(l1=l1),
            recurrent_regularizer=tf.keras.regularizers.L1(l1=l1),
        )
    ])

    model.compile(optimizer='adam', loss=loss_func(tau))
    return model
