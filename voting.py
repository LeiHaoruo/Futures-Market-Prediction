from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import lstm
import CNN
import MLP
import SVM
import XGBoost
import data_proc

def voting(pred_list):
	pred=pred_list.sum(1)
	pred[pred<3]=0
	pred[pred>2]=1
	return pred

def get_pred(X_test,X_group,Y_test):
	res_lstm = lstm.lstm_predict(X_test)
	res = res_lstm
	res_cnn = CNN.cnn_predict(X_test)
	res = np.hstack((res,res_cnn))
	res_mlp = MLP.mlp_predict(X_group)
	res = np.hstack((res,res_mlp))
	res_xgb = XGBoost.xgb_predict(X_group)
	res = np.hstack((res,res_xgb))
	res_svm = SVM.svm_predict(X_group)
	res = np.hstack((res,res_svm))

	np.save("res.npy",res)

	acu_list=[(res_lstm,"lstm"),(res_cnn,"cnn"),(res_mlp,"mlp"),(res_svm,"svm"),(res_xgb,"xgb")]
	for (pred,model) in acu_list:
		print model,": ",accuracy_score(Y_test, pred)



if __name__=='__main__':
	X_test,X_group,Y_test = data_proc.load_test("marketing_data/m0000/", [9])
	#get_pred(X_test,X_group,Y_test)
	res = np.load("res.npy")
	pred = voting(res)
	print accuracy_score(Y_test, pred)


