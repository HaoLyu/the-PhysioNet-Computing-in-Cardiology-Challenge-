class patient(object):

	def __init__(self,filename):
		self.filename = filename
		self.f = open('set-a/' + self.filename + '.txt', 'r')
	
	def readline(self):
		self.row = []
		for row in self.f.readlines():
			self.row.append(row)


if __name__ == '__main__':
	var_list = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
				"Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose",
				"HCO3", "HCT", "HR", "K", "Lactate", "Mg",
				"MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
				"PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
				"SysABP", "Temp", "TropI", "TropT", "Urine", "WBC",
				"Weight"]
	pa_dic = {}
	pa_dic["time"] = []
	for var in var_list:
		pa_dic[var] = []


	new = patient('132539')
	new.readline()
	n = 0
	for row in new.row:
		n = n+1
		if(n<7):
			continue

		pa_dic["time"].append(row.split(',')[0])
		pa_dic[row.split(',')[1]].append([row.split(',')[0],row.rstrip().split(',')[2]])

	print pa_dic