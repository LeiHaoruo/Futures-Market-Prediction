import pandas as pd
import numpy as np
from sklearn import preprocessing

attributes = ['InstrumentID', 'TradingDay', 'UpdateTime', 'UpdateMillisec', 'LastPrice', 'Volume', 'LastVolume', 'Turnover', 'LastTurnover', 'AskPrice5', 'AskPrice4', 'AskPrice3', 'AskPrice2', 'AskPrice1', 'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5', 'AskVolume5', 'AskVolume4', 'AskVolume3', 'AskVolume2', 'AskVolume1', 'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5', 'OpenInterest', 'UpperLimitPrice', 'LowerLimitPrice']
is_drop = [1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0]
TimeSlice = 20

def get_label(data,time):
	l = len(data)
	cur = time-1
	for i in range(cur+20,l):#1s or 20s
		if(data[cur][-1]<data[i][-1]):
			return 1  #up
		if(data[cur][-1]>data[i][-1]):
			return 0 #down
	return 0 #no change for all peroids

def load_data(path):
	df = pd.read_csv(path)
	for i in range(len(is_drop)):
		if(is_drop[i]==1):
			df=df.drop(attributes[i],1)
	df['PredictPrice']=(df['AskPrice1']+df['BidPrice1'])
	data = df.values
	data_ = np.delete(data,-1,axis=1)
	x_ =data_[0:TimeSlice]
	label= get_label(data,TimeSlice)
	y_ = np.array([label])
	for i in range(TimeSlice,len(data)-TimeSlice,TimeSlice):
		features = data_[i:i+TimeSlice]
		x_= np.vstack((x_,features))
		label= get_label(data,i+TimeSlice)
		y_ = np.append(y_,label)
	
	mm_scaler = preprocessing.MinMaxScaler()
	st_scaler = preprocessing.StandardScaler().fit(x_)	
	#x_ = mm_scaler.fit_transform(x_)	
	x_ = st_scaler.transform(x_)
	x_ = np.reshape(x_, (x_.shape[0]/20,20,x_.shape[1]))
	train_size = int(len(x_)*0.8)
	x_train = x_[0:train_size]
	y_train = y_[0:train_size]
	x_test = x_[train_size:]
	y_test = y_[train_size:]
	
	return x_train,y_train,x_test,y_test

def load_data_grouped(path):
	df = pd.read_csv(path)
	for i in range(len(is_drop)):
		if(is_drop[i]==1):
			df=df.drop(attributes[i],1)
	df['PredictPrice']=(df['AskPrice1']+df['BidPrice1'])
	data = df.values
	data_ = np.delete(data,-1,axis=1)
	x_ = np.ravel(data_[0:TimeSlice])
	label= get_label(data,TimeSlice)
	y_ = np.array([label])
	for i in range(TimeSlice,len(data)-TimeSlice,TimeSlice):
		features = np.ravel(data_[i:i+TimeSlice])
		x_= np.vstack((x_,features))
		label= get_label(data,i+TimeSlice)
		y_ = np.append(y_,label)
	
	mm_scaler = preprocessing.MinMaxScaler()
	st_scaler = preprocessing.StandardScaler().fit(x_)	
	#x_ = mm_scaler.fit_transform(x_)	
	x_ = st_scaler.transform(x_)
	train_size = int(len(x_)*0.8)
	x_train = x_[0:train_size]
	y_train = y_[0:train_size]
	x_test = x_[train_size:]
	y_test = y_[train_size:]
	
	return x_train,y_train,x_test,y_test