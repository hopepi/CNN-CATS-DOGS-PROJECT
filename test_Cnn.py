import tensorflow as tf
from keras.src.saving import load_model
from keras.src.utils.image_utils import load_img, img_to_array
import numpy as np
from pathlib import Path

model = load_model('my_model.h5')


base_dir = Path("C:/Users/umutk/OneDrive/Masaüstü/CatsAndDogs/my_Test")
img_path1 = base_dir / "test1.png"
img_path2 = base_dir / "test2.jpeg"
img_path3 = base_dir / "test3.jpeg"
img_path4 = base_dir / "test4.jpg"
img_path5 = base_dir / "test5.jpg"
img_path6 = base_dir / "test6.jpg"
img_path7 = base_dir / "test7.jpg"
img_path8 = base_dir / "test8.jpg"
img_path9 = base_dir / "test9.jpg"
img_path10 = base_dir / "test10.jpg"
img_path11 = base_dir / "test11.jpg"
img_path12 = base_dir / "test12.jpeg"
img_path13 = base_dir / "test13.jpg"
img_path14 = base_dir / "test14.jpeg"
img_path15 = base_dir / "test15.jpg"
img_path16 = base_dir / "test16.jpg"

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(180, 180))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

img_array1 = preprocess_image(img_path1)
img_array2 = preprocess_image(img_path2)
img_array3 = preprocess_image(img_path3)
img_array4 = preprocess_image(img_path4)
img_array5 = preprocess_image(img_path5)
img_array6 = preprocess_image(img_path6)
img_array7 = preprocess_image(img_path7)
img_array8 = preprocess_image(img_path8)
img_array9 = preprocess_image(img_path9)
img_array10 = preprocess_image(img_path10)
img_array11 = preprocess_image(img_path11)
img_array12 = preprocess_image(img_path12)
img_array13 = preprocess_image(img_path13)
img_array14 = preprocess_image(img_path14)
img_array15 = preprocess_image(img_path15)
img_array16 = preprocess_image(img_path16)


predictions1 = model.predict(img_array1)
predicted_class1 = np.argmax(predictions1, axis=1)
print(f'Test1 = Tahmin edilen sınıf: {predicted_class1}')

predictions2 = model.predict(img_array2)
predicted_class2 = np.argmax(predictions2, axis=1)
print(f'Test2 = Tahmin edilen sınıf: {predicted_class2}')

predictions3 = model.predict(img_array3)
predicted_class3 = np.argmax(predictions3, axis=1)
print(f'Test3 = Tahmin edilen sınıf: {predicted_class3}')

predictions4 = model.predict(img_array4)
predicted_class4 = np.argmax(predictions4, axis=1)
print(f'Test4 = Tahmin edilen sınıf: {predicted_class4}')

predictions5 = model.predict(img_array5)
predicted_class5 = np.argmax(predictions5, axis=1)
print(f'Test5 = Tahmin edilen sınıf: {predicted_class5}')

predictions6 = model.predict(img_array6)
predicted_class6 = np.argmax(predictions6, axis=1)
print(f'Test6 = Tahmin edilen sınıf: {predicted_class6}')

predictions7 = model.predict(img_array7)
predicted_class7 = np.argmax(predictions7, axis=1)
print(f'Test7 = Tahmin edilen sınıf: {predicted_class7}')

predictions8 = model.predict(img_array8)
predicted_class8 = np.argmax(predictions8, axis=1)
print(f'Test8 = Tahmin edilen sınıf: {predicted_class8}')

predictions9 = model.predict(img_array9)
predicted_class9 = np.argmax(predictions9, axis=1)
print(f'Test9 = Tahmin edilen sınıf: {predicted_class9}')

predictions10 = model.predict(img_array10)
predicted_class10 = np.argmax(predictions10, axis=1)
print(f'Test10 = Tahmin edilen sınıf: {predicted_class10}')

predictions11 = model.predict(img_array11)
predicted_class11 = np.argmax(predictions11, axis=1)
print(f'Test11 = Tahmin edilen sınıf: {predicted_class11}')

predictions12 = model.predict(img_array12)
predicted_class12 = np.argmax(predictions12, axis=1)
print(f'Test12 = Tahmin edilen sınıf: {predicted_class12}')

predictions13 = model.predict(img_array13)
predicted_class13 = np.argmax(predictions13, axis=1)
print(f'Test13 = Tahmin edilen sınıf: {predicted_class13}')

predictions14 = model.predict(img_array14)
predicted_class14 = np.argmax(predictions14, axis=1)
print(f'Test14 = Tahmin edilen sınıf: {predicted_class14}')

predictions15 = model.predict(img_array15)
predicted_class15 = np.argmax(predictions15, axis=1)
print(f'Test15 = Tahmin edilen sınıf: {predicted_class15}')

predictions16 = model.predict(img_array16)
predicted_class16 = np.argmax(predictions16, axis=1)
print(f'Test16 = Tahmin edilen sınıf: {predicted_class16}')

"""
%85 doğruluk
"""