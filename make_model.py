import numpy as np
import cv2, pickle
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.models import Model


# データファイルと画像サイズの指定
data_file = "./images\ETL8G.pickle"
im_size = 32
out_size = 12075 # あーんまでの文字の数
im_color = 3
in_shape = (im_size, im_size, im_color)

# 保存した画像データ一覧を読み込む
data = pickle.load(open(data_file, "rb"))

# 画像データを0-1の範囲に直す
y = []
x = []
for d in data:
  (num, img) = d
  img = img.astype('float').reshape(im_size, im_size, im_color) / 255
  y.append(keras.utils.np_utils.to_categorical(num, out_size))
  x.append(img)
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# vgg16のインスタンスの生成
input_tensor = Input(shape=(32, 32, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)


#モデル構築
top_model = Sequential()
'''
top_model.add(Conv2D(32, (3, 3), input_shape=in_shape))
top_model.add(BatchNormalization())
top_model.add(Activation('relu'))
top_model.add(MaxPooling2D(pool_size=(2, 2)))
top_model.add(Dropout(0.25))

top_model.add(Conv2D(64, (3, 3)))
top_model.add(BatchNormalization())
top_model.add(Activation('relu'))
top_model.add(Conv2D(64, (3, 3)))
top_model.add(BatchNormalization())
top_model.add(Activation('relu'))
top_model.add(MaxPooling2D(pool_size=(2, 2)))
top_model.add(Dropout(0.25))

top_model.add(Flatten())
top_model.add(Dense(512))
top_model.add(BatchNormalization())
top_model.add(Activation('relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(out_size))
top_model.add(Activation('softmax'))
'''

top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='softmax'))

# モデルの連結
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# vgg16の重みの固定
for layer in model.layers[:19]:
    layer.trainable = False

top_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


hist = top_model.fit(
  x_train, y_train,
  batch_size=64, epochs=50,verbose=1,
  validation_data=(x_test, y_test))

score = top_model.evaluate(x_test, y_test, verbose=1)
print("正解率 ", score[1], "loss ", score[0])

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

top_model.save('ETL8-model.h5')
