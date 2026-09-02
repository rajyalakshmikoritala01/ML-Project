import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import CottonDiseaseModel

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")

print("Initializing model...")
model = CottonDiseaseModel(model_path)
print("Predicting...")
prediction, probabilities = model.predict(img_path)
print("Prediction:", prediction)
print("Probabilities:", probabilities)
