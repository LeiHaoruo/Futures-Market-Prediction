from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import load_model
import numpy as np
import data_proc
import os

def lstm_with_generator(numFile):
    model = Sequential()
    #model.add(Embedding(max_features, 256, input_length=maxlen))
    model.add(LSTM(128, input_shape=(20,12)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    model.fit_generator(data_proc.generator_from_path("marketing_data/m0000/", file_set = [3,4,5,6,7,8]), steps_per_epoch=1970, epochs=15, validation_data =
         data_proc.generator_from_path("marketing_data/m0000/", [9]), validation_steps = 378)
    #res = model.evaluate_generator(data_proc.generator_from_path("marketing_data/m0000/", [14]), steps=300)
    #print ("score:",res[0])
    #print ("acc",res[1])

    model.save('lstm.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model

    # returns a compiled model
    # identical to the previous one
    model = load_model('lstm.h5')
    res = model.evaluate_generator(data_proc.generator_from_path("marketing_data/m0000/", [9]), steps=378)
    print ("score:",res[0])
    print ("acc",res[1])

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

def lstm_predict(X_test):
	model = load_model('lstm.h5')
	'''res = model.predict(X_test)
	label=model.predict_classes(X_test)
	res=np.hstack((res,label))'''
	res=model.predict_classes(X_test)
	return res


if __name__=='__main__':
	lstm_with_generator(5)
	#x_train,y_train,x_test,y_test = data_proc.load_data('marketing_data/m0000/TK_m0000[s20170404 00205000_e20170414 00153000]20170410_1755_46.csv')
	#lstm_train(x_train,y_train,x_test,y_test)
