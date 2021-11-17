import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


def main():
    training_data, test_data = fashion_mnist.load_data()
    (training_images, training_labels) = training_data
    (test_images, test_labels) = test_data
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    model = Sequential([
        Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        optimizer="adam",
    )

    model.fit(
        training_images,
        training_labels,
        epochs=50,
    )

    model.evaluate(test_images, test_labels)

    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


if __name__ == "__main__":
    main()
