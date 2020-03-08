import tensorflow as tf
import tensorflow_datasets as tf_ds


def train():
    # noinspection SpellCheckingInspection
    data_sets, information = tf_ds.load(name='mnist', with_info=True, as_supervised=True)
    # noinspection SpellCheckingInspection
    mnist_train, mnist_test = data_sets['train'], data_sets['test']

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_data = mnist_test.map(scale).batch(BATCH_SIZE)

    train_data = train_data.take(STEPS_PER_EPOCH)
    test_data = test_data.take(STEPS_PER_EPOCH)

    image_batch, label_batch = next(iter(train_data))
    print(image_batch)
    print(label_batch)

    image_batch, label_batch = next(iter(test_data))
    print(image_batch)
    print(label_batch)


BUFFER_SIZE = 10  # 실전 코드에서는 더 큰 값을 사용합니다.
BATCH_SIZE = 64
NUM_EPOCHS = 5
STEPS_PER_EPOCH = 5
