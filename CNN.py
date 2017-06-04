
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import numpy as np
import data_proc

def cnn_train(X_train,Y_train,X_test,Y_test):
	model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=(20, 12)))
	model.add(Conv1D(64, 3, activation='relu'))
	model.add(MaxPooling1D(3))
	model.add(Conv1D(128, 3, activation='relu'))
	model.add(Conv1D(128, 3, activation='relu'))
	model.add(GlobalAveragePooling1D())
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	model.fit(x_train, y_train, batch_size=16, epochs=10)
	score = model.evaluate(x_test, y_test, batch_size=16)
	print score

if __name__=='__main__':
	x_train,y_train,x_test,y_test = data_proc.load_data('TK_m0000[s20170404 00205000_e20170414 00153000]20170410_1755_46.csv')
	cnn_train(x_train,y_train,x_test,y_test)


