import numpy as np
import data_proc
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def xgboost_train(params={}):
	model = xgb.XGBClassifier(**params)

	train_iterators=data_proc.generator_from_path_group("marketing_data/m0000/", file_set = [3,4,5,6,7,8],svm=True)
	for i, (X_train, Y_train) in enumerate(train_iterators):
		model.fit(X_train,Y_train)
		joblib.dump(model, "xgboost.m")
	
	clf = joblib.load("xgboost.m")
	test_iterators=data_proc.generator_from_path_group("marketing_data/m0000/", file_set = [9],svm=True)
	for i, (X_test, Y_test) in enumerate(test_iterators):
		Y_pred = clf.predict(X_test)
		print(accuracy_score(Y_test, Y_pred))
	#model.fit(X_train, Y_train,eval_set=[(X_train, Y_train), (X_test, Y_test)], eval_metric='error')

def xgb_predict(X_test):
	clf = joblib.load("xgboost.m")
	res = clf.predict(X_test)
	return np.reshape(res,(res.shape[0],1))

if __name__=='__main__':
	#x_train,y_train,x_test,y_test = data_proc.load_data_grouped('TK_m0000[s20170404 00205000_e20170414 00153000]20170410_1755_46.csv')
	params = {'max_depth':3, 'subsample':1.0, 'min_child_weight':1.0,
              'colsample_bytree':1.0, 'learning_rate':0.1, 'silent':True}
	xgboost_train(params)
