from patient_level_time_var import patient
from os import listdir


outcomes = []
with open('Outcomes-a.txt', 'r') as f:
    outcomes = f.readlines()

RecordIDs = []
for f in listdir("./set-a"):
	RecordIDs.append(f)

for n in range(1,(len(RecordIDs))):
	test_patient = patient(RecordIDs[n])
	observation = test_patient.convert_to_time_value()
	print "survival patient status: ", outcomes[n]
	print "Patient's ID is: ",observation['RecordID']
	print "Temp is",observation['Temp']
	print '-'*50

