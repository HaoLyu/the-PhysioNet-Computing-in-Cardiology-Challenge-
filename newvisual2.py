from collections import OrderedDict
from math import log, sqrt
import numpy as np
import pandas as pd
from six.moves import cStringIO as StringIO
from bokeh.plotting import figure, show, output_file

patient_data = """
variables, survival, dead
Albumin, 1.1518906784, 1.49846613375        
ALP, 38.9586900847, 76.859215088      
ALT, 60.9184730984, 154.58952209          
AST, 76.8053950813, 270.996248782      
Bilirubin, 0.592279039881, 2.21461456364             
BUN, 23.277702843, 36.7815788103         
Cholesterol,   11.8743470691, 12.3393501805   
Creatinine, 1.28057115689, 1.76910688564    
DiasABP, 41.4940156671, 42.7322274708         
FiO2,  0.360537675575, 0.450833056257      
GCS,  11.7345551978, 9.45314906374          
Glucose,    131.434187701, 145.995310115
HCO3, 23.3801901005, 22.109043158
HCT, 30.9088795786, 31.0957918405
HR, 85.268220221, 89.0124949418
K, 4.0239520255, 4.10960690633
Lactate, 1.12826875867, 2.12307788422
Mg, 1.95452307462, 2.03935531827
MAP,  56.2342735863, 58.704807735
MechVent, 0.619268717353, 0.712996389892
Na, 136.0427718, 137.578854689
NIDiasABP, 50.2164518916, 48.2115363125
NIMAP, 66.6528324997, 64.8306405855
NISysABP,  102.659746579, 100.894294256
PaCO2, 30.0883484763, 33.1956542309
PaO2, 111.170123841, 116.016003702
pH, 5.57261049715, 6.6080495173
Platelets,  202.366719846, 198.46288881
RespRate, 5.74687351483, 3.32674242862
SaO2, 42.8388292682, 45.88767696
SysABP, 82.5726623659, 85.5441223263
Temp, 36.3556487346, 36.5423427493
TroponinI, 0.287830403781, 0.788402527076
TroponinT,  0.209555996601, 0.356216305656
Urine, 135.808379279, 94.5775743826
WBC, 12.0448733094, 13.9684298764
Weight, 77.0327811579, 76.0586113727
Height, 90.9883052815, 79.4696750903
Age, 63.3273360418, 69.9711191336
"""


patient_color = OrderedDict([
    ("Survival",   "#c64737"),
    ("Dead",     "#0d3362"  ),
])


df = pd.read_csv(StringIO(patient_data),
                 skiprows=1,
                 skipinitialspace=True,
                 engine='python')

width = 800
height = 800
inner_radius = 90
outer_radius = 300 - 10

minr = sqrt(log(.001 * 1E4))
maxr = sqrt(log(1000 * 1E4))
a = (outer_radius - inner_radius) / (minr - maxr)
b = inner_radius - a * maxr

def rad(mic):
    return a * np.sqrt(np.log(mic * 1E4)) + b

big_angle = 2.0 * np.pi / (len(df) + 1)
small_angle = big_angle / 7

x = np.zeros(len(df))
y = np.zeros(len(df))

output_file("burtin.html", title="average variables of survival and dead patients")

p = figure(plot_width=width, plot_height=height, title="",
    x_axis_type=None, y_axis_type=None,
    x_range=[-420, 420], y_range=[-420, 420],
    min_border=0, outline_line_color="black",
    background_fill="#f0e1d2", border_fill="#f0e1d2")

p.line(x+1, y+1, alpha=0)

# annular wedges
angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle

p.annular_wedge(
    x, y, inner_radius, outer_radius, -big_angle+angles, angles, color='#e69584'
)

# small wedges
p.annular_wedge(x, y, inner_radius, rad(df.survival),
    -big_angle+angles+5*small_angle, -big_angle+angles+6*small_angle,
    color=patient_color['Survival'])

p.annular_wedge(x, y, inner_radius, rad(df.dead),
    -big_angle+angles+1*small_angle, -big_angle+angles+2*small_angle,
    color=patient_color['Dead'])

# circular axes and lables
labels = np.power(10.0, np.arange(-3, 4))
radii = a * np.sqrt(np.log(labels * 1E4)) + b
p.circle(x, y, radius=radii, fill_color=None, line_color="white")
p.text(x[:-1], radii[:-1], [str(r) for r in labels[:-1]],
    text_font_size="8pt", text_align="center", text_baseline="middle")

# radial axes
p.annular_wedge(x, y, inner_radius-10, outer_radius+10,
    -big_angle+angles, -big_angle+angles, color="black")

# variables labels
xr = radii[0]*np.cos(np.array(-big_angle/2 + angles))
yr = radii[0]*np.sin(np.array(-big_angle/2 + angles))
label_angle=np.array(-big_angle/2+angles)
label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
p.text(xr, yr, df.variables, angle=label_angle,
    text_font_size="9pt", text_align="center", text_baseline="middle")

p.rect([-40, -40, ], [18, 0], width=30, height=13,
    color=list(patient_color.values()))
p.text([-15, -15], [18, 0], text=list(patient_color.keys()),
    text_font_size="9pt", text_align="left", text_baseline="middle")

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

show(p)