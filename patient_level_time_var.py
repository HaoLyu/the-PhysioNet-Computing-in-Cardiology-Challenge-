# Convert the raw data into better forms
class patient(object):

	def __init__(self,filename):
		self.filename = filename
		self.f = open('set-a/' + self.filename, 'r')
		self.row = []
		for row in self.f.readlines():
			self.row.append(row)

	def convert_to_time_value(self):
		var_list = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
				"Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose",
				"HCO3", "HCT", "HR", "K", "Lactate", "Mg",
				"MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
				"PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
				"SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
				"Weight", "ICUType", "Height", "Gender", "Age", "RecordID"]

		pa_dic = {}
		pa_dic["time"] = []
		for var in var_list:
			pa_dic[var] = []

		for row in self.row[1:]:
			pa_dic["time"].append(row.split(',')[0])
			pa_dic[row.split(',')[1]].append([row.split(',')[0],row.rstrip().split(',')[2]])

		return pa_dic

if __name__ == '__main__':
	
	new = patient('132539')
	print new.row
	result = new.convert_to_time_value()
	

	print result