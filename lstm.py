from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import numpy as np
import data_proc

def lstm_train(X_train,Y_train,X_test,Y_test):
	model = Sequential()
	#model.add(Embedding(max_features, 256, input_length=maxlen))
	model.add(LSTM(128, input_shape=(20,12)))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

	model.fit(X_train, Y_train, batch_size=16, epochs=15,validation_data=(X_test,Y_test))
	res = model.evaluate(X_test, Y_test, batch_size=16)
	print ("score:",res[0])
	print ("acc",res[1])

if __name__=='__main__':
	x_train,y_train,x_test,y_test = data_proc.load_data('TK_m0000[s20170404 00205000_e20170414 00153000]20170410_1755_46.csv')
	lstm_train(x_train,y_train,x_test,y_test)
