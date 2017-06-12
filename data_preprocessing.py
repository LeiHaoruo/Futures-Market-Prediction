import os
import data_proc
import pandas as pd

path = 'marketing_data/m0000/'
name = 'file_'

for parent, dirnames, filenames in os.walk(path):
    for f_csv in filenames:
	print "reading file: "+f_csv
	index = f_csv[f_csv.rfind('_')+1:f_csv.rfind('.')]
	newFile = path+name+index+'.csv'
	if os.path.exists(newFile):
		print "ignore:"+newFile
		continue
	if int(index)==10:
		x,y,m,n = data_proc.load_data(path+f_csv,True)
		result = pd.DataFrame({'label':y})
		result.to_csv(path+name+index+'.csv')
		print y.shape,"write to",path+name+index+'.csv',"done"
        
