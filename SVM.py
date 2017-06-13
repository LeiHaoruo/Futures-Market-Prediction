import numpy as np
import data_proc
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

'''use sgd with hinge loss to do svm on large data'''

def svm_train():

	model = SVC(verbose=True)
	train_iterators=data_proc.generator_from_path_group("marketing_data/m0000/", file_set = [2,3,4],svm=True)
	for i, (X_train, Y_train) in enumerate(train_iterators):
		model.fit(X_train,Y_train)
		joblib.dump(model, "svm1.m")
	
	clf = joblib.load("svm1.m")
	test_iterators=data_proc.generator_from_path_group("marketing_data/m0000/", file_set = [9],svm=True)
	for i, (X_test, Y_test) in enumerate(test_iterators):
		Y_pred = clf.predict(X_test)
		print(accuracy_score(Y_test, Y_pred))

def svm_predict(X_test):
	clf = joblib.load("svm.m")
	res = clf.predict(X_test)
	return np.reshape(res,(res.shape[0],1))

def svm_sgd():
	sgd_clf = SGDClassifier(loss="hinge", penalty="l2")
	minibatch_train_iterators = data_proc.generator_from_path_group("marketing_data/m0000/", file_set = [1,2,3,4,5,6],sgd=True)

	for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
	    sgd_clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))
	    print("{} time".format(i))  
	    #print("{} score".format(sgd_clf.score(X_test, y_test))) 
	joblib.dump(sgd_clf, "svm.m")

	clf = joblib.load("svm.m")
	test_iterators = data_proc.generator_from_path_group("marketing_data/m0000/", file_set = [7,8],test=True)
	for i, (X_test, y_test) in enumerate(test_iterators):
		Y_pred = clf.predict(X_test)
		print(accuracy_score(y_test, Y_pred))



if __name__=='__main__':
	'''
	x_train,y_train,x_test,y_test = data_proc.load_data_grouped('TK_m0000[s20170404 00205000_e20170414 00153000]20170410_1755_46.csv')
	svm_train(x_train,y_train,x_test,y_test)
	'''
	#svm_sgd()
	svm_train()