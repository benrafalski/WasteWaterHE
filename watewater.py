import pandas as pd 
import tenseal as ts
import numpy as np

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
    flows = df[['Flow (L)']].drop_duplicates()

    area_conc = [concentrations[i*7:(i+1)*7] for i in range(6)]
    area_flows = [flows[i*7:(i+1)*7] for i in range(6)]

    week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    area_conc = {week_days[i] : area_conc[i] for i in range(6)}
    area_flows = {week_days[i] : area_flows[i] for i in range(6)}

    return (area_conc, area_flows)

def context():
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 40]	
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = pow(2, 40)
    ctx.generate_galois_keys()
    return ctx

# make note for the runtime for each party

file = 'Practice Data.xlsx'
area_conc, area_flows = read_data(file)
context = context()


# Mesa Input: (B) Flow Mesa [B_M1, B_M2]
# Tempe's Input: (D) Flow Tempe [D_T1, D_T2]
# Third-party Input: (A) Heroin Mesa, (C) Heroin Tempe
# Output: C*D-A*B
# Extra: (E) Heroin Guadalupe, (F) Flow Guadalupe

# Area 1: [(C_T1*D_T1) - (A_M1*B_M1)] - [A_M2*(B_M2*0.15)] - (C_DT*D_DT) - (C_CR*D_CR)
# Area 2: (C_T2*D_T2) - (C_S*D_S)
# Area 3: [(C_T3*D_T3) - [A_M2*(B_M2*0.85)]] - (E-F)


C_T1 = np.array([42.28])
D_T1 = np.array([58376509.76])
A_M1 = np.array([70.94])
B_M1 = np.array([17810352.47])
A_M2 = np.array([56.78])
B_M2 = np.array([43427157.25])
C_DT = np.array([54.79])
D_DT = np.array([3355764.17])
C_CR = np.array([60.76])
D_CR = np.array([4180563.65])
C_T2 = np.array([26.80])
D_T2 = np.array([17539221.87])
C_S = np.array([16.78])
D_S = np.array([2742756.14])
C_T3 = np.array([50.29])
D_T3 = np.array([50824570.13])
E = np.array([62.08])
F = np.array([946350.00])
CONST1 = np.array([0.85])
CONST2 = np.array([0.15])



x = np.array([1.1, 2.2, 3.3])
y = np.array([3.3, 2.2, 1.1])

x_enc = ts.ckks_vector(context, x)
y_enc = ts.ckks_vector(context, y)



C_T1_ENC = ts.ckks_vector(context, C_T1)
print(C_T1_ENC.decrypt())
D_T1_ENC = ts.ckks_vector(context, D_T1)
print(D_T1_ENC.decrypt())
A_M1_ENC = ts.ckks_vector(context, A_M1)
B_M1_ENC = ts.ckks_vector(context, B_M1)
A_M2_ENC = ts.ckks_vector(context, A_M2)
B_M2_ENC = ts.ckks_vector(context, B_M2)
C_DT_ENC = ts.ckks_vector(context, C_DT)
D_DT_ENC = ts.ckks_vector(context, D_DT)
C_CR_ENC = ts.ckks_vector(context, C_CR)
D_CR_ENC = ts.ckks_vector(context, D_CR)
C_T2_ENC = ts.ckks_vector(context, C_T2)
D_T2_ENC = ts.ckks_vector(context, D_T2)
C_S_ENC = ts.ckks_vector(context, C_S)
D_S_ENC = ts.ckks_vector(context, D_S)
C_T3_ENC = ts.ckks_vector(context, C_T3)
D_T3_ENC = ts.ckks_vector(context, D_T3)
E_ENC = ts.ckks_vector(context, E)
F_ENC = ts.ckks_vector(context, F)
CONST1_ENC = ts.ckks_vector(context, CONST1)
CONST2_ENC = ts.ckks_vector(context, CONST2)

area1 = (C_T1*D_T1) - (A_M1*B_M1) - (A_M2*(B_M2*0.15)) - (C_DT*D_DT) - (C_CR*D_CR)
area1_enc = (C_T1_ENC*D_T1_ENC) - (A_M1_ENC*B_M1_ENC) - (A_M2_ENC*(B_M2_ENC*CONST2_ENC)) - (C_DT_ENC*D_DT_ENC) - (C_CR_ENC*D_CR_ENC)
print(area1)
print(area1_enc.decrypt())


area2 = (C_T2*D_T2) - (C_S*D_S)
area2_enc = (C_T2_ENC*D_T2_ENC) - (C_S_ENC*D_S_ENC)
print(area2)
print(area2_enc.decrypt())


area3 = ((C_T3*D_T3) - (A_M2*(B_M2*CONST1))) - (E-F)
area3_enc = ((C_T3_ENC*D_T3_ENC) - (A_M2_ENC*(B_M2_ENC*CONST1_ENC))) - (E_ENC-F_ENC)
print(area3)
print(area3_enc.decrypt())





