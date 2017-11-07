import csv
import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

################### models ##################################

def baseline(model):
	model.add(Flatten())
	model.add(Dense(1))
	return model

def lenet(model):
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(16,5,5,activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model

def nvidia(model):
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
	model.add(Convolution2D(64,3,3,activation='relu'))
	model.add(Convolution2D(64,3,3,activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

################################################################

samples = []
with open('e:/sample-data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2, shuffle=True)

augmentation_factor = 6 # each sample is 3 images (center, left, right) x 2 (reversed)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size] # grab batch of samples

			images = []
			inputs = []

			for batch_sample in batch_samples:
				for i in range(3):
					image_path = 'e:/sample-data/' + batch_sample[i].lstrip()
					image = cv2.imread(image_path)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					images.append(image)
				offset = 0.2
				input = float(batch_sample[3])
				inputs.append(input)
				inputs.append(input + offset)
				inputs.append(input - offset)

			augmented_images = []
			augmented_inputs = []
			for image, input in zip(images, inputs):
				augmented_images.append(image)
				augmented_inputs.append(input)
				flipped_image = cv2.flip(image, 1)
				flipped_input = input * -1.0
				augmented_images.append(flipped_image)
				augmented_inputs.append(flipped_input)

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_inputs)
			yield (X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

#model = baseline(model)
#model = lenet(model)
model = nvidia(model)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=augmentation_factor*len(train_samples),
	validation_data=validation_generator, nb_val_samples=augmentation_factor*len(validation_samples),
	nb_epoch=5)

model.save('model.h5')
