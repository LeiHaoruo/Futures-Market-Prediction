import numpy as np
import data_proc
import xgboost as xgb

def xgboost_train(X_train,Y_train,X_test,Y_test,params={}):
	model = xgb.XGBClassifier(**params)
	model.fit(X_train, Y_train,eval_set=[(X_train, Y_train), (X_test, Y_test)], eval_metric='error')


if __name__=='__main__':
	x_train,y_train,x_test,y_test = data_proc.load_data_grouped('TK_m0000[s20170404 00205000_e20170414 00153000]20170410_1755_46.csv')
	params = {'max_depth':3, 'subsample':1.0, 'min_child_weight':1.0,
              'colsample_bytree':1.0, 'learning_rate':0.1, 'silent':True}
	xgboost_train(x_train,y_train,x_test,y_test,params)
