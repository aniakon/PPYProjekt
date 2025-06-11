import cv2
import numpy as np
from tensorflow.keras.models import load_model

def tryOut(photo):
    """
    Function allows to test the prediction of the model on the given photo.
    Args:
        photo (str): path to the photo
    Writes:
        prediction (List[int]): predictions of the model
        predicted class (int)
    """
    img = cv2.imread(photo)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))

    model = load_model("hand_model_finetuned.keras")

    img = img.astype('float32')
    img_batch = np.expand_dims(img, axis=0)

    pred = model.predict(img_batch)
    predicted_class = np.argmax(pred, axis=1)[0]
    print(pred)
    print(predicted_class)

tryOut("test0_11.jpg")