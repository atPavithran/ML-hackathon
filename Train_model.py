import os
import cv2
import numpy as np
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras

harmful_dir = os.path.join('Dataset','Harmful')
harmless_dir = os.path.join('Dataset','Harmless')

harmful_model_path = os.path.join('Dataset','harmful_dataset')
harmless_model_path = os.path.join('Dataset','harmless_dataset')

new_size = (224, 224)

def processed_img(directory):
    images = []
    labels = []
    for file in os.listdir(directory):
        if file.endswith('.jpg') or file.endswith('.png'):  
            img_path = os.path.join(directory, file)
            label = 1 if directory == harmful_dir else 0
            image = cv2.imread(img_path)
            if image is None:
                pass
            else:
                image = cv2.resize(image, new_size) 
                image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)

x_harmful, y_harmful = processed_img(harmful_dir)
x_harmless, y_harmless = processed_img(harmless_dir)


x_data = np.concatenate((x_harmful, x_harmless), axis=0)
y_data = np.concatenate((y_harmful, y_harmless), axis=0)


x_data = x_data / 255.0
split_idx = int(0.8 * len(x_data))
x_train, y_train = x_data[:split_idx], y_data[:split_idx]
x_test, y_test = x_data[split_idx:], y_data[split_idx:]


model_path = os.path.join("E:\HX\Model3")
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification, so using sigmoid activation
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

predictions = model.predict(x_test)

model.save('E:\HX\Dataset\model2.h5')

