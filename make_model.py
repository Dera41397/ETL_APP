from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2, pickle
import keras
import matplotlib.pyplot as plt

# データファイルと画像サイズの指定
data_file = "./images\ETL8G.pickle"
im_size = 32
out_size = 12075 # あーんまでの文字の数
im_color = 1
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

#モデル構築
model = Sequential()

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(out_size))
model.add(Activation('softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer= RMSprop(),
  metrics=['accuracy'])

hist = model.fit(
  x_train, y_train,
  batch_size=64, epochs=50,verbose=1,
  validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
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
