# Get the average value of each varibale of survival patients and dead patients
from patient_level_time_var import patient
from os import listdir
import numpy as np
import pandas as pd

# Read data and outcomes
outcomes = []
with open('Outcomes-a.txt', 'r') as f:
    outcomes = f.readlines()

RecordIDs = []
for f in listdir("./set-a"):
	RecordIDs.append(f)

# Get the list of dead and survival patients
dead_list = []
survival_list = []

for n in range(1,(len(RecordIDs))):

	if(outcomes[n].rstrip().split(',')[5] == '1'):
		dead_list.append(n)
	else:
		survival_list.append(n)

var_list = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
				"Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose",
				"HCO3", "HCT", "HR", "K", "Lactate", "Mg",
				"MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
				"PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
				"SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
				"Weight", "Height", "Age"]

# Get the average vlaue of specific variable for one patient
def get_patient_value(ID, var):
	test_patient = patient(ID)
	observation = test_patient.convert_to_time_value()
	if len(observation[var]) == 0:
		return 0

	average_list = []

	for n in range(0,len(observation[var])):
		value = float(observation[var][n][1])

		if(value < 0):
			continue 
		else:
			average_list.append(value)

	if len(average_list) == 0:
		return 0

	average = sum(average_list) / float(len(average_list))
	return average

# average value of each variable of dead patients
for var in var_list:
	var_ave = []

	for id in dead_list:
		ID = RecordIDs[id]
		var_ave.append(get_patient_value(ID, var))

	print "variable is:", var, "and the average is:", sum(var_ave) / float(len(var_ave))

# average value of each variable of survival patients
for var in var_list:
	var_ave = []

	for id in survival_list:
		ID = RecordIDs[id]
		var_ave.append(get_patient_value(ID, var))

	print "variable is:", var, "and the average is:", sum(var_ave) / float(len(var_ave))
