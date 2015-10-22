import os
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import random

variables = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
			 "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose",
			 "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
			 "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
			 "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
			 "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
			 "Weight", "Height", "Gender", "Age", "ICUType"]

def load_file(filename):
	with open(filename) as f:
		reader = csv.reader(f, delimiter=',')
		header = next(reader)
		d = {}
		for var in variables:
			d[var] = []
		_, _, idx = next(reader) # dont treat id like rest of variables
		d['RecordID'] = idx
		for row in reader:
			time, variable, value = row
			value = float(value)
			d[variable].append(value)
	return d
	
def load_results(filename):
	with open(filename) as f:
		reader = csv.reader(f, delimiter=',')
		header = next(reader)
		d = {}
		for row in reader:
			idx, saps, sofa, len_stay, survival, in_hos_death = row
			d[idx] = int(in_hos_death)
	return d
	
def to_vec(x):
	vec = []
	for var in variables[:-1]:
		vec.append(x[var])
	if x['ICUType'] == 1.0:
		vec += [0.0, 0.0, 0.0]
	elif x['ICUType'] == 2.0:
		vec += [1.0, 0.0, 0.0]
	elif x['ICUType'] == 3.0:
		vec += [0.0, 1.0, 0.0]
	else:
		vec += [0.0, 0.0, 1.0]
	return vec
	
def to_disc_vec(x, disc_vars):
	vec = []
	for var in disc_vars:
		if var != 'ICUType':
			vec.append(x[var])
	if x['ICUType'] == 1.0:
		vec += [1]#[0.0, 0.0, 0.0]
	elif x['ICUType'] == 2.0:
		vec += [2]#[1.0, 0.0, 0.0]
	elif x['ICUType'] == 3.0:
		vec += [3]#[0.0, 1.0, 0.0]
	else:
		vec += [4]#[0.0, 0.0, 1.0]
	return vec
	
def to_cont_vec(x, cont_vars):
	vec = []
	for var in cont_vars:
		vec.append(x[var])
	return vec
		
def mytype(list):

	if list[0]>list[1]:
		return 0
	else:
		return 1


def leftcompare(list1, list2):
	n = 0
	for i in range(0, len(list1)):
		if(list1[i]==list2[i])&(list1[i]==1):
			n = n + 1

	return float(n)/float(len(list1))

def get_data():
	files = os.listdir('set-a')
	#random.shuffle(files)
	X = []
	Y = []
	results = load_results('Outcomes-a.txt')
	k = 0
	l = 0
	for filename in files:
		x = load_file('set-a/' + filename)
		idx = x['RecordID']
		y = results[idx]
		X.append(x)
		Y.append(y)
	print(k)
	print(l)
	msng = {}
	for var in variables:
		k = 0
		n = 0
		for x,y in zip(X,Y):
			if x[var] != [] and x[var] != [-1.0]:
				n += 1
				if y == 1:
					k += 1
		msng[var] = k * 1.0 / n
	#for k,v in msng.iteritems():
	#    print(k + ',' + str(v))
	d = {}
	for var in variables:
		d[var] = 0
	for x in X:
		for var in variables:
			if x[var] == [] or x[var] == [-1.0]:
				d[var] += 1
	var_to_indicator = []
	# get missing item info
	for var, val in d.items():
		v = val * 1.0 / len(X)
		if v >= 0.75:
			var_to_indicator.append(var)
	var_to_avg = [var for var in variables if var not in var_to_indicator]
	avg = {}
	for var in var_to_avg:
		avg[var] = []
		for x in X:
			if x[var] != [] and x[var] != [-1.0]:
				xavg = sum(x[var]) / len(x[var])
				avg[var].append(xavg)
		avg[var] = sum(avg[var]) / len(avg[var])
	Xcont = []
	Xdisc = []
	for i,x in enumerate(X):
		for var in var_to_indicator:
			if x[var] == [] or x[var] == [-1.0]:
				x[var] = 0
			else:
				x[var] = 1
		for var in var_to_avg:
			if x[var] == [] or x[var] == [-1.0]:
				x[var] = avg[var]
			else:
				xavg = sum(x[var]) / len(x[var])
				x[var] = xavg
		X[i] = np.array(to_vec(x))
		Xcont.append(np.array(to_cont_vec(x, var_to_avg)))
		Xdisc.append(np.array(to_disc_vec(x, var_to_indicator)))
	X = np.array(X)
	Xcont = np.array(Xcont)
	Xdisc = np.array(Xdisc)
	Y = np.array(Y)
	kf = cross_validation.KFold(len(X), n_folds=3)
	print('running models')
	qual = []
	meansc = []
	meansd = []
	meansf = []
	mean_left_score = []
	for train_index, test_index in kf:
		#model = LogisticRegression()
		#model = KNeighborsClassifier(n_neighbors=10)
		#model = RandomForestClassifier()
		#model2 = AdaBoostClassifier()
		#model = svm.LinearSVC()
		#model = svm.SVC(kernel='poly', degree=2)
		modeld = MultinomialNB()
		modelc = GaussianNB()
		#model = GradientBoostingClassifier(n_estimators=200)
		#model = GaussianNB()
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		
		
		Xcont_train, Xcont_test = Xcont[train_index], Xcont[test_index]
		#model.fit(Xcont_train, Y_train)
		
		k = len(X) // 2
		Xcont_train1 = Xcont_train[:k]
		Xcont_train2 = Xcont_train[k:]
		Xdisc_train, Xdisc_test = Xdisc[train_index], Xdisc[test_index]
		Xdisc_train1 = Xdisc_train[:k]
		Xdisc_train2 = Xdisc_train[k:]
		Y_train1 = Y_train[:k]
		Y_train2 = Y_train[k:]
		#model.fit(X_train, Y_train)
		modelc.fit(Xcont_train1, Y_train1)
		modeld.fit(Xdisc_train1, Y_train1)
		
		predictc_train = modelc.predict_proba(Xcont_train2)
		predictd_train = modeld.predict_proba(Xdisc_train2)
		train2 = np.hstack((predictc_train, predictd_train))
		modelf = GaussianNB()
		modelf.fit(train2, Y_train2)
		
		predictc_test = modelc.predict_proba(Xcont_test)
		predictd_test = modeld.predict_proba(Xdisc_test)

		test_pairs = np.hstack((predictc_test, predictd_test))

		predict = modelf.predict(test_pairs)

		predict_proba = modelf.predict_proba(test_pairs)

		for number in range(0,len(predict_proba)):
			predict_proba[number] = [mytype(predict_proba[number]), predict_proba[number][1]]


		#print '-'*50
		#print 'predict result:', predict_proba
		#print '-'*50
####	
		meansc.append(modelc.theta_)
		meansd.append(modeld.coef_)
		meansf.append(modelf.theta_)

		leftpre = []
		leftreal =[]

		# set the number of top patients we will use to get the LEFT score
		leftsetting = 400
	


		order = []
		for i,y in enumerate(predict_proba):
			order.append([i,y[1],y[0]])

		order = sorted(order,reverse=True, key=lambda x: x[1])
	
		leftnumber = []
		#print '*'*50

		print "length of order is:", len(order)
		for hao in range(0, leftsetting):
			#print order[hao]
			leftpre.append(order[hao][2])
			leftnumber.append(order[hao][0])
		#print '*'*50
		for hao in range(0, leftsetting):
			#print Y_test[leftnumber[hao]]
			leftreal.append(Y_test[leftnumber[hao]])
		#print '*'*50
		print "In one fold, the predict of top is: ", leftcompare(leftpre, leftreal)

		print 'len(Y_test)', len(Y_test)
		print 'sum of Y_test',sum(Y_test)
		sample_rate = float(sum(Y_test))/float(len(Y_test))

		left_score = leftcompare(leftpre, leftreal)/sample_rate

		print "left_socre is :", left_score
		mean_left_score.append(left_score)
####
		#model.fit(X_train, Y_train)
		#predict = model.predict(X_test)
		TP = 0
		FP = 0
		FN = 0
		TN = 0
		for i,y in enumerate(predict):
			truth = Y_test[i]
			if (truth == 0):
				if y == 0:
					TN += 1
				else:
					FN += 1
			else:
				if y == 0:
					FP += 1
				else:
					TP += 1
		Se = TP * 1.0 / (TP + FN)
		plusP = TP * 1.0 / (TP + FP)
		qual.append(min(Se,plusP))
	print'qual: ', (qual)
	print(sum(qual) / 3)
	print "mean left score is",(sum(mean_left_score) / 3)
	print(meansc[0])
	print(meansd[0])
	print(meansf[0])

def bar_graph(filename):
	X = []
	Y = []
	with open(filename) as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			var, val = row
			val = float(val)
			X.append(var)
			Y.append(val)
	ind = np.arange(len(Y)) + 0.75
	width = 0.45
	fig, ax = plt.subplots(1,1)
	p = ax.bar(ind, Y, width)
	ax.set_xticks (ind + width/2.0)
	ax.set_xticklabels( X, rotation = 70 )
	ax.axhline(0.135, color='black', linewidth=2)
	ax.set_ylabel('Fraction of Patients with Measurements who Passes Away')
	fig.tight_layout()
	plt.show()
		
			
	
	
if __name__ == '__main__':
	get_data()
			
		
			