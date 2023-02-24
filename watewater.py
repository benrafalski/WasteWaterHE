import time
import pandas as pd 
import tenseal as ts
import numpy as np
import sys
'''
Input (data format, etc)
Mesa heroin concentration (ng/L) (collected from T)
Total Mesa wastewater flow data over 24 hrs (L per day)
Tempe heroin concentration (ng/L) (collected from S)
Total Tempe flow data flow data over 24 hrs (L per day)

How to compute the result (formula) 
(Heroin concentration Tempe * Flow Tempe) - (Heroin concentration Mesa * Flow Mesa) 
'''




def read_data(file):
    df = pd.read_excel(file)
    
    concentrations = df[["Avg (ng/L)"]].dropna()    
    flows = df[['Flow (L)']]

    area_conc = [concentrations[i*7:(i+1)*7] for i in range(6)]
    area_flows = [flows[i*7:(i+1)*7] for i in range(6)]

    week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    area_conc = {week_days[i] : area_conc[i] for i in range(6)}
    area_flows = {week_days[i] : area_flows[i] for i in range(6)}

    return (area_conc, area_flows)

file = 'Practice Data.xlsx'
df = pd.read_excel(file)

concentrations = df[["Avg (ng/L)"]].dropna()    
flows = df[['Flow (L)']]
area_conc = [list(concentrations[i*7:(i+1)*7].values.flat) for i in range(11)]
area_flows = [list(flows[i*14:(i+1)*14].values.flat) for i in range(11)]
for i in range(len(area_flows)): del area_flows[i][1::2]

# A and C = concentration
# B and D = flow

C_T1 = np.array(area_conc[0])
D_T1 = np.array(area_flows[0])
A_M1 = np.array(area_conc[6])
B_M1 = np.array(area_flows[6])
A_M2 = np.array(area_conc[7])
B_M2 = np.array(area_flows[7])
C_DT = np.array(area_conc[9])
D_DT = np.array(area_flows[9])
C_CR = np.array(area_conc[10])
D_CR = np.array(area_flows[10])
C_T2 = np.array(area_conc[1])
D_T2 = np.array(area_flows[1])
C_S = np.array(area_conc[8])
D_S = np.array(area_flows[8])
C_T3 = np.array(area_conc[2])
D_T3 = np.array(area_conc[2])
E = np.array(area_conc[5])
F = np.array(area_flows[5])
CONST1 = np.array([0.85]*7)
CONST2 = np.array([0.15]*7)


def context():
    poly_mod_degree = 8192*2
    coeff_mod_bit_sizes = [60, 40, 40, 40, 60]		
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = pow(2, 40)
    ctx.generate_galois_keys()
    return ctx

# make note for the runtime for each party

# file = 'Practice Data.xlsx'
# area_conc, area_flows = read_data(file)
ctx = context()


# Mesa Input: (B) Flow Mesa [B_M1, B_M2]
# Tempe's Input: (D) Flow Tempe [D_T1, D_T2]
# Third-party Input: (A) Heroin Mesa, (C) Heroin Tempe
# Output: C*D-A*B
# Extra: (E) Heroin Guadalupe, (F) Flow Guadalupe

# Area 1: [(C_T1*D_T1) - (A_M1*B_M1)] - [A_M2*(B_M2*0.15)] - (C_DT*D_DT) - (C_CR*D_CR)
# Area 2: (C_T2*D_T2) - (C_S*D_S)
# Area 3: [(C_T3*D_T3) - [A_M2*(B_M2*0.85)]] - (E-F)


# C_T1 = np.array([42.28])
# D_T1 = np.array([58376509.76])
# A_M1 = np.array([70.94])
# B_M1 = np.array([17810352.47])
# A_M2 = np.array([56.78])
# B_M2 = np.array([43427157.25])
# C_DT = np.array([54.79])
# D_DT = np.array([3355764.17])
# C_CR = np.array([60.76])
# D_CR = np.array([4180563.65])
# C_T2 = np.array([26.80])
# D_T2 = np.array([17539221.87])
# C_S = np.array([16.78])
# D_S = np.array([2742756.14])
# C_T3 = np.array([50.29])
# D_T3 = np.array([50824570.13])
# E = np.array([62.08])
# F = np.array([946350.00])
# CONST1 = np.array([0.85])
# CONST2 = np.array([0.15])



x = np.array([1.1, 2.2, 3.3])
y = np.array([3.3, 2.2, 1.1])

x_enc = ts.ckks_vector(ctx, x)
y_enc = ts.ckks_vector(ctx, y)


print(x*y)
print((x_enc*y_enc).decrypt())


np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:0.3f}'.format})
C_T1_ENC = ts.ckks_vector(ctx, C_T1)
D_T1_ENC = ts.ckks_vector(ctx, D_T1)
A_M1_ENC = ts.ckks_vector(ctx, A_M1)
B_M1_ENC = ts.ckks_vector(ctx, B_M1)
A_M2_ENC = ts.ckks_vector(ctx, A_M2)
B_M2_ENC = ts.ckks_vector(ctx, B_M2)
C_DT_ENC = ts.ckks_vector(ctx, C_DT)
D_DT_ENC = ts.ckks_vector(ctx, D_DT)
C_CR_ENC = ts.ckks_vector(ctx, C_CR)
D_CR_ENC = ts.ckks_vector(ctx, D_CR)
C_T2_ENC = ts.ckks_vector(ctx, C_T2)
D_T2_ENC = ts.ckks_vector(ctx, D_T2)
C_S_ENC = ts.ckks_vector(ctx, C_S)
D_S_ENC = ts.ckks_vector(ctx, D_S)
C_T3_ENC = ts.ckks_vector(ctx, C_T3)
D_T3_ENC = ts.ckks_vector(ctx, D_T3)
E_ENC = ts.ckks_vector(ctx, E)
F_ENC = ts.ckks_vector(ctx, F)
CONST1_ENC = ts.ckks_vector(ctx, CONST1)
CONST2_ENC = ts.ckks_vector(ctx, CONST2)


start = time.time()
area1 = (C_T1*D_T1) - (A_M1*B_M1) - (A_M2*(B_M2*0.15)) - (C_DT*D_DT) - (C_CR*D_CR)
area1_enc = (C_T1_ENC*D_T1_ENC) - (A_M1_ENC*B_M1_ENC) - (A_M2_ENC*(B_M2_ENC*CONST2_ENC)) - (C_DT_ENC*D_DT_ENC) - (C_CR_ENC*D_CR_ENC)
print(f'\nArea 1 Expected: {area1.tolist()}')
print(f'Area 1 Actual:   {area1_enc.decrypt()}')


area2 = (C_T2*D_T2) - (C_S*D_S)
area2_enc = (C_T2_ENC*D_T2_ENC) - (C_S_ENC*D_S_ENC)
print(f'\nArea 2 Expected: {area2.tolist()}')
print(f'Area 2 Actual:   {area2_enc.decrypt()}')


area3 = ((C_T3*D_T3) - (A_M2*(B_M2*CONST1))) - (E-F)
area3_enc = ((C_T3_ENC*D_T3_ENC) - (A_M2_ENC*(B_M2_ENC*CONST1_ENC))) - (E_ENC-F_ENC)
print(f'\nArea 3 Expected: {area3.tolist()}')
print(f'Area 3 Actual:   {area3_enc.decrypt()}')

print(f'\nComputation time: {(time.time()-start)*1000} ms')





