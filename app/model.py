import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASS_MAP = {
    0: "Alternaria Leaf Spot",
    1: "Bacterial Blight",
    2: "Fusarium Wilt",
    3: "Healthy Cotton Leaf",
    4: "Verticillium Wilt"
}

class CottonDiseaseModel:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, img_path):

        img = image.load_img(img_path, target_size=(224,224))
        img = image.img_to_array(img)
        img = img/255.0
        img = np.expand_dims(img, axis=0)

        preds = self.model.predict(img)[0]

        class_index = np.argmax(preds)

        predicted_class = CLASS_MAP[class_index]

        probabilities = {
            CLASS_MAP[i]: float(preds[i]) for i in range(len(preds))
        }

        return predicted_class, probabilities