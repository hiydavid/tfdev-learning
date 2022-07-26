# test tensorflow on local machine
import tensorflow as tf
from tensorflow.keras import datasets, layers

# check version
print(tf.__version__)

# get data
print("Getting MNIST images...")
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# normalize images
print("Normalizing images...")
train_images, test_images = train_images / 255.0, test_images / 255.0

# build model
print("Building & compiling model...")
model = tf.keras.Sequential([
    layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.Flatten(),
    layers.Dense(10, activation="softmax")
])

# compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# fit model
print("Training model...")
model.fit(
    x=train_images,
    y=train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)

# evaluate model
print("Evaluating model...")
model.evaluate(test_images, test_labels)

# Save model to current working directory
model.save("test_image_model.h5")
