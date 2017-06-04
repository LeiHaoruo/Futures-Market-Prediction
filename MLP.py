from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import data_proc

def mlp_train(X_train,Y_train,X_test,Y_test):
	model = Sequential()
	model.add(Dense(64, input_dim=240, activation='relu'))
	model.add(Dropout(0.5))#
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	model.fit(x_train, y_train,
	          epochs=20,
	          batch_size=128)
	score = model.evaluate(x_test, y_test, batch_size=128)
	res = model.evaluate(X_test, Y_test, batch_size=16)
	print ("score:",res[0])
	print ("acc",res[1])

if __name__=='__main__':
	x_train,y_train,x_test,y_test = data_proc.load_data_grouped('TK_m0000[s20170404 00205000_e20170414 00153000]20170410_1755_46.csv')
	mlp_train(x_train,y_train,x_test,y_test)


