from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.models import load_model
import numpy as np
import data_proc

def mlp_train():
	model = Sequential()
	model.add(Dense(64, input_dim=240, activation='relu'))
	model.add(Dropout(0.5))#
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	model.fit_generator(data_proc.generator_from_path_group("marketing_data/m0000/", file_set = [3,4,5,6,7,8]), steps_per_epoch=1970, epochs=15, validation_data =
         data_proc.generator_from_path_group("marketing_data/m0000/", [9]), validation_steps = 378)
	model.save('mlp.h5')  # creates a HDF5 file 'cnn.h5'
	del model  # deletes the existing model

	model = load_model('mlp.h5')
	res = model.evaluate_generator(data_proc.generator_from_path_group("marketing_data/m0000/", [9]), steps=378)
	print ("score:",res[0])
	print ("acc",res[1])

	'''model.fit(x_train, y_train,
	          epochs=20,
	          batch_size=128)
	score = model.evaluate(x_test, y_test, batch_size=128)
	res = model.evaluate(X_test, Y_test, batch_size=16)
	print ("score:",res[0])
	print ("acc",res[1])'''

def mlp_predict(X_test):
	model = load_model('mlp.h5')
	res = model.predict_classes(X_test)
	return res

if __name__=='__main__':
	#x_train,y_train,x_test,y_test = data_proc.load_data_grouped('TK_m0000[s20170404 00205000_e20170414 00153000]20170410_1755_46.csv')
	mlp_train()


