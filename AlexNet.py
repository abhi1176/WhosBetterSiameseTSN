
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
		Dense, Dropout, Activation, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

def AlexNet():
	AlexNet = Sequential()
	#1st Convolutional Layer
	AlexNet.add(Conv2D(filters=96, input_shape=(224,224,3),
					   kernel_size=(11,11), strides=(4,4), padding='same',
					   activation='relu'))
	AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

	#2nd Convolutional Layer
	AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1),
					   padding='same', activation='relu'))
	AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

	#3rd Convolutional Layer
	AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
					   padding='same', activation='relu'))

	#4th Convolutional Layer
	AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
					   padding='same', activation='relu'))

	#5th Convolutional Layer
	AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),
					   padding='same', activation='relu'))
	AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

	AlexNet.add(Flatten())

	# 1st Fully Connected Layer
	AlexNet.add(Dense(4096, input_shape=(32,32,3,), activation='relu'))
	AlexNet.add(Dropout(0.5))

	#2nd Fully Connected Layer
	AlexNet.add(Dense(4096, activation='relu'))
	AlexNet.add(Dropout(0.5))

	#3rd Fully Connected Layer
	AlexNet.add(Dense(1000, activation='relu'))
	AlexNet.add(Dropout(0.5))

	#Output Layer
	AlexNet.add(Dense(1, activation='sigmoid'))

	#Model Summary
	# AlexNet.summary()
	return AlexNet

