"""
Test olarak yapılmıştır DEĞERLERİ CİDDİYE ALMAYIN MODELİN AŞIRI ÖĞRENMESİ VAR
"""

import tensorflow as tf
from keras import layers
from keras.src.models import Model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.api.applications import EfficientNetB0
from pathlib import Path

num_cores = 3
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

base_dir = Path("C:/Users/umutk/OneDrive/Masaüstü/CatsAndDogs")
train_dir = base_dir / "train"
test_dir = base_dir / "test"
validation_dir = base_dir / "validation"


def create_dataframe(directory):
    filepaths = []
    labels = []

    for label_dir in directory.iterdir():
        if label_dir.is_dir():
            for img_file in label_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    filepaths.append(str(img_file))
                    labels.append(label_dir.name)

    if not filepaths or not labels:
        raise ValueError("Filepaths or labels are empty.")
    if len(filepaths) != len(labels):
        raise ValueError("Filepaths and labels must have the same length.")

    data = {'Filepath': filepaths, 'Label': labels}
    df = pd.DataFrame(data)
    return df

train_df = create_dataframe(train_dir)
test_df = create_dataframe(test_dir)
validation_df = create_dataframe(validation_dir)


train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, fill_mode="nearest")
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    color_mode='rgb',
    target_size=(180, 180),
    class_mode='sparse',
    batch_size=64,
    shuffle=True,
    seed=0,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(180, 180),
    class_mode='sparse',
    batch_size=64
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(180, 180),
    class_mode='sparse',
    batch_size=64
)


# EfficientNetB0 modelini oluşturma
# Modelin ağırlıklarını ImageNet veri setinden alıyoruz.
# include_top=False ile modelin sonundaki tam bağlantılı katmanları dahil etmiyoruz
# çünkü kendi sınıflandırma katmanlarımızı yapıcaz.
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

# Yeni model oluşturma
# Modelin giriş boyutu
# inputs değişkeni, modelin giriş katmanı için bir şekil tanımlar.
inputs = layers.Input(shape=(180, 180, 3))

# Rescaling katmanı ekleme
# Bu katman, görüntü değerlerini 0-255 aralığından 0-1 aralığına ölçeklendirir.
# Görüntü verilerinin normalizasyonu, modelin daha iyi öğrenmesini sağlar.
x = layers.Rescaling(1./255)(inputs)

# Temel EfficientNetB0 modelini ekleme
# training=False ifadesi, temel modelin ağırlıklarının eğitim sırasında güncellenmeyeceği anlamına gelir.
# Bu, transfer öğrenme durumlarında kullanılır.
x = base_model(x, training=False)

# Global Average Pooling katmanı ekleme
# Bu katman, giriş görüntülerinin her bir kanalını ortalamak için kullanılır.
# Özellikle, modelin daha iyi genelleme yapmasını ve aşırı öğrenmeyi azaltmasını sağlar.
x = layers.GlobalAveragePooling2D()(x)
"""
# Yeni Dense katmanı ekleme
# Burada iki yeni Dense katmanı ekliyoruz ve Dropout katmanı ile birlikte kullanıyoruz.
# İlk Dense katmanı, 128 nöron içerir ve ReLU aktivasyon fonksiyonu kullanır.
# Bu, modelin daha fazla özellik öğrenmesini sağlayacak.
x = layers.Dense(128, activation='relu')(x)  # 128 nöronlu bir gizli katman ekliyoruz.
"""

x = layers.Dropout(0.5)(x)

# Çıkış katmanı
# units=1 parametresi, çıkış katmanının 1 nöron (birlikte ikili sınıflandırma) içereceğini belirtir.
# activation='sigmoid' ise çıkış değerinin 0-1 aralığında olacağını ve
# ikili sınıflandırma problemlerinde (örneğin, kedi veya köpek) kullanılacağını belirtir.
outputs = layers.Dense(1, activation='softmax')(x)  # Çıkış katmanı, sigmoid aktivasyonu.

# Modeli tanımlama
# inputs ve outputs ile yeni model oluşturuluyor.
# Bu, tüm katmanları bir araya getirerek bir model oluşturur.
model = Model(inputs, outputs)

# Modeli derleme
# Modelin eğitim sürecinde kullanılacak optimizasyon algoritması, kayıp fonksiyonu ve metrikler belirleniyor.
# optimizer="adam", Adam optimizasyon algoritmasını kullanır.
# Bu algoritma, öğrenme hızını otomatik olarak ayarlayarak daha hızlı bir şekilde eğitim sağlar.
# loss="binary_crossentropy" ifadesi, ikili sınıflandırma problemleri için kullanılan kayıp fonksiyonudur.
# Bu fonksiyon, modelin tahminleri ile gerçek değerler arasındaki farkı ölçer.
# metrics=['accuracy']` ifadesi ise modelin performansını değerlendirmek için doğruluk (accuracy) metriklerini kullanır.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history  = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=3
)

# Modeli değerlendirme
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
model.save('EfficientNet_my_model.h5')
"""
%70
Aşırı Öğrenme
"""
