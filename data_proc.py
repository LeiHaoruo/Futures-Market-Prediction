import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
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

#is_fit: true for training, false for testing
#file_set: index of files we use
def generator_from_path(path,file_set):
    steps = 32
    count = 0
    prefix_inx = 1
    f_csv_list = []
    for parent, dirnames, filenames in os.walk(path):
        for p_inx in file_set:
	    for f in filenames:
		if f[f.rfind('_')+1:f.rfind('.')]==str(p_inx) and f.find('TK')>=0:
	            f_csv_list.append(f)
	print f_csv_list
	while 1:
	    for f_csv in f_csv_list:
	        print "reading file: "+f_csv
	        X,Y,x_none,y_none = load_data(path+f_csv,True)
	        print "processing file: "+f_csv
	        for i in range(0,Y.shape[0]-steps,steps):
	            yield(X[i:i+steps],Y[i:i+steps])

def load_test(path,file_set):
    f_csv_list = []
    for parent, dirnames, filenames in os.walk(path):
        for p_inx in file_set:
	    for f in filenames:
		if f[f.rfind('_')+1:f.rfind('.')]==str(p_inx) and f.find('TK')>=0:
	            f_csv_list.append(f)
	print f_csv_list
	X_train=[]
	X_group=[]
	Y_train=[]
	for f_csv in f_csv_list:
		print "reading file: "+f_csv
		X_g,Y,x_none,y_none = load_data_grouped(path+f_csv,True)
		X,Y,x_none,y_none = load_data(path+f_csv,True)
		print "processing file: "+f_csv
		if X_train == []:
			X_train = X
			Y_train = Y
			X_group = X_g
			continue
		X_train=np.vstack((X_train,X))
		Y_train=np.vstack((Y_train,Y))
		X_group=np.vstack((X_group,X_g))
	return X_train,X_group,Y_train

	
	
#no_separate: True for no separate it into train and test parts
def load_data(path,no_separate):
	df = pd.read_csv(path)
	for i in range(len(is_drop)):
		if(is_drop[i]==1):
			df=df.drop(attributes[i],1)
	df['PredictPrice']=(df['AskPrice1']+df['BidPrice1'])
	data = df.values
	data_ = np.delete(data,-1,axis=1)
	'''
	x_ =data_[0:TimeSlice]
	label= get_label(data,TimeSlice)
	y_ = np.array([label])
	for i in range(TimeSlice,len(data)-TimeSlice,TimeSlice):
		features = data_[i:i+TimeSlice]
		x_= np.vstack((x_,features))
		label= get_label(data,i+TimeSlice)
		y_ = np.append(y_,label)
	'''
	index = path[path.rfind('_')+1:path.rfind('.')]
	dirs = path[0:path.rfind('/')+1]
	y_ = pd.read_csv(dirs+'file_'+index+'.csv')
	y_ = y_.values
	y_ = np.delete(y_,0,axis=1)
	#print "y element:",y_[:10]
	#mm_scaler = preprocessing.MinMaxScaler()
	st_scaler = preprocessing.StandardScaler().fit(data_)
	x_ = st_scaler.transform(data_)
	d_size = x_.shape[0]/TimeSlice*TimeSlice
	x_ = np.reshape(x_[0:d_size], (x_.shape[0]/TimeSlice,TimeSlice,x_.shape[1]))
	bias=x_.shape[0]-y_.shape[0]
	if bias>0:
		x_=x_[:-bias]
	print "x's shape",x_.shape, "y's shape",y_.shape	
	#x_ = mm_scaler.fit_transform(x_)	
	#x_ = st_scaler.transform(x_)
	#x_ = np.reshape(x_, (x_.shape[0]/20,20,x_.shape[1]))
	train_size = int(len(x_)*1)
	if no_separate:
	    return x_,y_,None,None
	x_train = x_[0:train_size]
	y_train = y_[0:train_size]
	x_test = x_[train_size:]
	y_test = y_[train_size:]
	
	return x_train,y_train,x_test,y_test

def generator_from_path_group(path,file_set,svm=False,sgd=False,test=False):
    steps = 32
    steps_2=500
    count = 0
    prefix_inx = 1
    f_csv_list = []
    for parent, dirnames, filenames in os.walk(path):
        for p_inx in file_set:
	    for f in filenames:
		if f[f.rfind('_')+1:f.rfind('.')]==str(p_inx) and f.find('TK')>=0:
	            f_csv_list.append(f)
	print f_csv_list

	if test == True:
		for f_csv in f_csv_list:
			print "reading file: "+f_csv
			X,Y,x_none,y_none = load_data_grouped(path+f_csv,True)
			print "processing file: "+f_csv
			yield(np.array(X),np.array(Y))

	elif svm == True:
		X_train=[]
		Y_train=[]
		for f_csv in f_csv_list:
			print "reading file: "+f_csv
			X,Y,x_none,y_none = load_data_grouped(path+f_csv,True)
			print "processing file: "+f_csv
			if X_train == []:
				X_train = X
				Y_train = Y
				continue
			X_train=np.vstack((X_train,X))
			Y_train=np.vstack((Y_train,Y))
		yield(np.array(X_train),np.array(Y_train).ravel())

	elif sgd == True:
		for f_csv in f_csv_list:
			print "reading file: "+f_csv
			X,Y,x_none,y_none = load_data_grouped(path+f_csv,True)
			print "processing file: "+f_csv
			for i in range(0,Y.shape[0]-steps_2,steps_2):
				yield(np.array(X[i:i+steps_2]),np.array(Y[i:i+steps_2]))

	else:		
		while 1:
		    for f_csv in f_csv_list:
		        print "reading file: "+f_csv
		        X,Y,x_none,y_none = load_data_grouped(path+f_csv,True)
		        print "processing file: "+f_csv
		        for i in range(0,Y.shape[0]-steps,steps):
		            yield(X[i:i+steps],Y[i:i+steps])
		


def load_data_grouped(path,no_separate):
	df = pd.read_csv(path)
	for i in range(len(is_drop)):
		if(is_drop[i]==1):
			df=df.drop(attributes[i],1)
	df['PredictPrice']=(df['AskPrice1']+df['BidPrice1'])
	data = df.values
	data_ = np.delete(data,-1,axis=1)

	index = path[path.rfind('_')+1:path.rfind('.')]
	dirs = path[0:path.rfind('/')+1]
	y_ = pd.read_csv(dirs+'file_'+index+'.csv')
	y_ = y_.values
	y_ = np.delete(y_,0,axis=1)
	#print "y element:",y_[:10]
	#mm_scaler = preprocessing.MinMaxScaler()
	st_scaler = preprocessing.StandardScaler().fit(data_)
	x_ = st_scaler.transform(data_)
	d_size = x_.shape[0]/TimeSlice*TimeSlice
	x_ = np.reshape(x_[0:d_size], (x_.shape[0]/TimeSlice,TimeSlice*x_.shape[1]))
	bias=x_.shape[0]-y_.shape[0]
	if bias>0:
		x_=x_[:-bias]
	print "x's shape",x_.shape, "y's shape",y_.shape	
	#x_ = mm_scaler.fit_transform(x_)	
	#x_ = st_scaler.transform(x_)
	#x_ = np.reshape(x_, (x_.shape[0]/20,20,x_.shape[1]))
	train_size = int(len(x_)*1)
	if no_separate:
	    return x_,y_,None,None
	x_train = x_[0:train_size]
	y_train = y_[0:train_size]
	x_test = x_[train_size:]
	y_test = y_[train_size:]
	
	return x_train,y_train,x_test,y_test
