import cv2
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model('E:\HX\model1.h5')
directory = 'E:\\HX\\test_images'
count = 1

for file in os.listdir(directory):
    if file.endswith('.jpg') or file.endswith('.png'): 
        img_path = os.path.join(directory, file)
        image = cv2.imread(img_path)
        new_size = (224,224)
        image = cv2.resize(image,new_size) 
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image_array = np.array(image)
        image_array = np.expand_dims(image_array , axis = 0)

        predictions = model.predict(image_array)
        print('The',count,'objects prediction is:')
        print(predictions[0])







