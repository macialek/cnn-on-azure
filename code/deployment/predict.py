import os
import logging
import json
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory.
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.getenv("AZUREML_MODEL_DIR")

    # load TF model into memory
    model = tf.keras.models.load_model(os.path.join(model_path, 'model'))
    logger.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.

    """
    logger.info("Request received")
    try:
        data = json.loads(raw_data)
    except Exception as e:
        message = f'Unable to process input JSON data: {e}'
        logger.exception(message)
        return {'exception': message}

    image_path = tf.keras.utils.get_file('image', origin=data['image_url'])

    img = tf.keras.utils.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    answer = {'image_url': data['image_url'], 'class': 'np.argmax(score)', 'probability': 100 * np.max(score)}
    logger.info("Request processed")
    return answer
