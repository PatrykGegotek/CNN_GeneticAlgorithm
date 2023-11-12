import tensorflow as tf


def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label


def load_and_scale_images(data_dir, target_size=(64, 64), batch_size=128):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        label_mode='binary',
        labels='inferred',
        color_mode='rgb',
        image_size=target_size,
        batch_size=batch_size
    )
    return test_ds.map(process)
