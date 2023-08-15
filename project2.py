from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D

classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dense(units=4, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('C:/Users/15593/OneDrive/Documents/IT298/Fruits_quality_check/data1/Data2/train',
                                                 target_size=(100, 100),
                                                 batch_size=6,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory('C:/Users/15593/OneDrive/Documents/IT298/Fruits_quality_check/data1/Data2/test',
                                            target_size=(100, 100), 
                                            batch_size=6,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         samples_per_epoch=763,
                         nb_epoch=20,
                         validation_data=test_set,
                         validation_steps=448)

classifier.save("model.h9")