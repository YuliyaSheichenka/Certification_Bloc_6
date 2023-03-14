import tensorflow as tf


def image2tensor(image_as_bytes, dim=(224, 224)):
    """
    Receives a image as bytes as input, that will be loaded,
    preprocessed and turned into a Tensor.
    """

    # Apply the same preprocessing as during training (resize and rescale)
    image = tf.io.decode_image(image_as_bytes, channels=3)
    image = tf.image.resize(image, dim)
    image = image/255.

    # Convert the Tensor to a batch of Tensors
    image = tf.expand_dims(image, 0)
    return image


labels = {
    0: 'NonDemented',
    1: 'VeryMildDemented',
    2: 'MildDemented',
    3: 'ModerateDemented'
}


def pred2label(prediction):
    prediction = tf.argmax(prediction).numpy()
    return labels[prediction]
