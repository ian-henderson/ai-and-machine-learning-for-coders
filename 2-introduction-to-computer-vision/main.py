import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Flatten


class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


def main():
    # Fashion MNIST is designed to have 60,000 training images and 10,000 test
    # images. So, the return from fashion_mnist.load_data() will give you an
    # array of 60,000 28x28-pixel arrays called training_images, and an array
    # of 60,000 values (0-9) called training_labels. Similarly, the
    # test_images array will contain 10,000 28x28-pixel arrays, and the
    # test_labels array will contain 10,000 values between 0 and 9.
    training_data, test_data = fashion_mnist.load_data()
    (training_images, training_labels) = training_data
    (test_images, test_labels) = test_data
    # NumPy allows you to do an operation across the entire array with this
    # notation. All of the Fashion MNIST images are grayscale, with values
    # between 0 and 255. Dividing by 255 ensures that every pixel is
    # represented by a number between 0 and 1 instead. This process is called
    # normalizing the image.
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
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
        callbacks=[MyCallback()],
        epochs=50,
    )

    model.evaluate(test_images, test_labels)

    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


if __name__ == "__main__":
    main()
