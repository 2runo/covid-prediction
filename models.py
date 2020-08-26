"""
모델을 빌드한다.
"""
import tensorflow as tf
import options as opt


def build_model():
    # 모델을 반환한다.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(opt.SIGHT, 1)),
        tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(16, dropout=0.2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(opt.Y_N, activation=tf.keras.layers.LeakyReLU()),
    ])

    optim = tf.keras.optimizers.Adam(lr=opt.lr)
    model.compile(optimizer=optim, loss="mse")  # regression에 유리한 MSE loss 사용

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
