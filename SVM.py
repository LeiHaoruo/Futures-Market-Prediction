import numpy as np
import data_proc
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

def svm_train(X_train,Y_train,X_test,Y_test,params={}):
	model = SVC()
	model.fit(X_train,Y_train)
	Y_pred = model.predict(X_test)
	print(accuracy_score(Y_test, Y_pred))

if __name__=='__main__':
	x_train,y_train,x_test,y_test = data_proc.load_data_grouped('TK_m0000[s20170404 00205000_e20170414 00153000]20170410_1755_46.csv')
	svm_train(x_train,y_train,x_test,y_test)
