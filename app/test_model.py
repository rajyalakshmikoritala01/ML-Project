from model import CottonDiseaseModel

# load trained model
model = CottonDiseaseModel("model.h5")

# test image
image_path = "test.jpg"
prediction, probabilities = model.predict(image_path)

print("\n===== Cotton Disease Prediction =====")

print("\nPredicted Disease:", prediction)

print("\nProbabilities:")
for disease, prob in probabilities.items():
    print(f"{disease} : {prob:.4f}")
    
