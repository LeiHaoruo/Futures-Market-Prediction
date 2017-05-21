from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

attributes = ['InstrumentID', 'TradingDay', 'UpdateTime', 'UpdateMillisec', 'LastPrice', 'Volume', 'LastVolume', 'Turnover', 'LastTurnover', 'AskPrice5', 'AskPrice4', 'AskPrice3', 'AskPrice2', 'AskPrice1', 'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5', 'AskVolume5', 'AskVolume4', 'AskVolume3', 'AskVolume2', 'AskVolume1', 'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5', 'OpenInterest', 'UpperLimitPrice', 'LowerLimitPrice']
is_drop = [1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0]
TimeSlice = 20

max_features = 850*7
maxlen = 260

def get_label(data):
	l = int(len(data)/2)
	for i in range(l):
		if(data[i][-1]<data[i+TimeSlice][-1]):
			return 1,0   #up
		if(data[i][-1]>data[i+TimeSlice][-1]):
			return 0,0   #down
	return 0,1  #no change for all peroids

def load_data(path):
	cnt=0
	df = pd.read_csv(path)
	for i in range(len(is_drop)):
		if(is_drop[i]==1):
			df=df.drop(attributes[i],1)
	df['PredictPrice']=(df['AskPrice1']+df['BidPrice1'])
        data = df.values
	print ("before process",np.shape(data))
	x_ = np.ravel(data[0:TimeSlice])
	label,flag = get_label(data[TimeSlice:3*TimeSlice])
	cnt+=flag
	y_ = np.array([label])
	for i in range(0,len(data)-2*TimeSlice,TimeSlice):
		if i:
			features = np.ravel(data[i:i+TimeSlice])
			x_ = np.vstack((x_,features))
			label,flag = get_label(data[i+TimeSlice:i+3*TimeSlice])
			cnt+=flag
			y_ = np.append(y_,label)
	
	print np.shape(x_),np.shape(y_)
	print cnt
	mm_scaler = preprocessing.MinMaxScaler()	
	x_ = mm_scaler.fit_transform(x_)	
	
	train_size = int(len(data)*0.7)
	x_train = x_[0:train_size]
	y_train = y_[0:train_size]
	x_test = x_[train_size:]
	y_test = y_[train_size:]
	
	return x_train,y_train,x_test,y_test
def trainModel(X_train,Y_train,X_test,Y_test):
	model = Sequential()
	model.add(Embedding(max_features, 256, input_length=maxlen))
	model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

	model.fit(X_train, Y_train, batch_size=16, epochs=10,validation_data=(X_test,Y_test))
	score,acc = model.evaluate(X_test, Y_test, batch_size=16)
	print ("score:",score)
	print ("acc",acc)

if __name__=='__main__':
	x_train,y_train,x_test,y_test = load_data('m0000/read_m0.csv')
	trainModel(x_train,y_train,x_test,y_test)
