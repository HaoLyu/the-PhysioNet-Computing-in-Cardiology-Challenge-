from patient_level_time_var import patient
from os import listdir
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file
import datetime
import random

# Convert string date time to datetime.datetime type
def timeconvert(time):
	year = 2012
	month = 1
	day = 1
	hour = int(time.split(":")[0])
	minute = int(time.split(":")[1])

	if (hour>23)&(hour<48):
		day = 2
		hour = int(hour - 24)
	elif hour == 48:
		day = 3
		hour = 0

	newtime = datetime.datetime(year, month, day, hour, minute)
	return newtime

# Read data and outcomes
outcomes = []
with open('Outcomes-a.txt', 'r') as f:
    outcomes = f.readlines()

RecordIDs = []
for f in listdir("./set-a"):
	RecordIDs.append(f)

# Randomly pick 7 survival patients and 1 dead patient
dead_list = []
live_list = []

for n in range(1,(len(RecordIDs))):

	if(outcomes[n].rstrip().split(',')[5] == '1'):
		dead_list.append(n)
	else:
		live_list.append(n)

random_list =[]
random_list.append(random.sample(dead_list,1)[0])

for j in range(0,7):
	random_list.append(random.sample(live_list,7)[j])  

# Plot the line graph showing time and temperature of 7 random survival patients and 1 dead one
output_file("color_scatter.html", title="patient_Temp_time.py example")
p = figure(width=800, height=250, x_axis_type="datetime", title="Temperature and Timeseries")
p.xaxis.axis_label = "Time"
p.yaxis.axis_label = "Temp"

for i in random_list:
	test_patient = patient(RecordIDs[i])
	observation = test_patient.convert_to_time_value()
	data=[]
	
	# Dead in hospital is marked by red line
	# Survival is marked by navy line
	if(outcomes[i].rstrip().split(',')[5]=='1'):
		linecolor = 'red'
		linelegend = 'dead'
	else:
		linecolor = 'navy'
		linelegend = 'survival'

	for n in range(0,len(observation['Temp'])):
		observe_time = timeconvert(observation['Temp'][n][0])
		observe_temp = float(observation['Temp'][n][1])
		data.append((observe_time, observe_temp))

	data_array = np.array(data)
	p.line(data_array[:,0], data_array[:,1], color=linecolor, alpha=0.5, legend = linelegend)

p.legend.label_text_font = "times"
p.legend.orientation = "top_left"
show(p)