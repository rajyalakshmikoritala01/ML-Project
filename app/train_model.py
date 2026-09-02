import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import json
DATASET_PATH= r"C:\Users\RAJIII KORITALA\OneDrive\Desktop\cotton-plant-disease-prediction - real\Cotton Disease Dataset\Cotton_Augmented_Dataset"
train_datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)
print("Class Indices:", train_data.class_indices)
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128,activation="relu"),
    Dense(5,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
model.save("model.h5")

print("✅ Model training completed!")