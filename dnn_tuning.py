import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

class DNN_Tuning:

    def __init__(self, train_data, test_data):
        self.train_data = train_data.copy().sample(frac=0.8, random_state=0)
        self.test_data = test_data.copy()
        self.train_labels = self.train_data.pop('SalePrice')
        self.test_labels = self.test_data.pop('SalePrice')

    def plot_loss(self, history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 200000])
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        plt.show()

    def normalizer(self):
        # data = self.clean_data(False)
        #train_data = train_data.transpose()
        normalizer = layers.Normalization(axis=-1)
        normalizer = layers.Normalization()
        normalizer.adapt(self.train_data.to_numpy())
        
        print(normalizer.mean.numpy())
        return normalizer

    def model_builder(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten())

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units = hp.Int('units', min_value=32, max_value=512, step=16)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))
        model.add(keras.layers.Dense(9))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss='mean_absolute_error',
                        metrics=['accuracy'])

        return model

    def build_dnn_model(self, norm) -> keras.Sequential:
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))

        return model

    def dnn_model(self):
        dnn_model = build_dnn_model(normalizer())
        dnn_model.summary()
        history = dnn_model.fit(
            self.train_data,
            self.train_labels,
            validation_split=0.2,
            verbose=0, epochs=10000)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail(10))

        plot_loss(history)
        self.test_results = self.dnn_model.evaluate(self.test_data, self.test_labels, verbose=0)
        print(self.test_results)
        
        self.test_predictions = self.dnn_model.predict(self.test_data).flatten()
    
    def kt_tuner(self):
        tuner = kt.Hyperband(self.model_builder,
                     objective='val_accuracy',
                     max_epochs=500,
                     factor=10,
                     hyperband_iterations=5)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(self.train_data, self.train_labels, epochs=1000, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
                self.train_data,
                self.train_labels,
                validation_split=0.2,
                verbose=0, epochs=5000)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))


        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail(20))

        self.plot_loss(history)
        test_results = model.evaluate(self.test_data, self.test_labels, verbose=0)
        print(test_results)


if __name__ == "__main__":
    
    training_data = pd.DataFrame(
            pd.read_csv("house-prices-advanced-regression-techniques/train_copy.csv"))
    test_data = pd.DataFrame(
            pd.read_csv("house-prices-advanced-regression-techniques/test_copy.csv"))
    dnn_tuning = DNN_Tuning(training_data, test_data)
    dnn_tuning.kt_tuner()