import tensorflow as tf
import keras_tuner as kt

import matplotlib.pyplot as plt

(img_train, label_train), (img_test, label_test) = \
    tf.keras.datasets.fashion_mnist.load_data()

print('Train: X=%s, y=%s' % (img_train.shape, label_train.shape))
print('Test: X=%s, y=%s' % (img_test.shape, label_test.shape))

# Plot small sample
fig, axs = plt.subplots(3, 3)
[ax.imshow(img_train[idx],
           cmap='binary')
 for idx, ax in enumerate(axs.ravel())]
[ax.axis('off') for ax in axs.ravel()]  # Remove axis labels
plt.savefig("examples.pdf", bbox_inches='tight', dpi=1200)


def model_builder(hp):
    """ The model builder function returns a compiled model and uses
    hyperparameters you define inline to hypertune the model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(img_train, label_train, epochs=50, validation_split=0.2,
             callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first 
densely-connected layer is {best_hps.get('units')} and the optimal learning rate 
for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2,
                    callbacks=[stop_early])

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print(f'Best epoch: {best_epoch}')
