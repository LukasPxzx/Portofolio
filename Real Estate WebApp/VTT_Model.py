from scipy.integrate import cumtrapz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load Original Dataset
data = pd.read_csv('/Users/lukassspazo/Year 3 Python/Nasa_UK_21-22.csv')

#Adjust temperatures for December to February
data['Date'] = pd.to_datetime(data['Date'])

#Define the RCP scenario adjustments
percentage_of_century = 2 / 80

#RCP8.5
temp_increase_8_5 = 4.3 * percentage_of_century
humidity_increase_8_5 = temp_increase_8_5 * 5  #Assuming a 5% increase in humidity for each degree increase in temperature

#RCP4.5
temp_increase_4_5 = 2.5 * percentage_of_century
humidity_increase_4_5 = temp_increase_4_5 * 5

#RCP2.6 (assuming minimal changes)
temp_increase_2_6 = 0
humidity_increase_2_6 = 0

#Create a dataset for each RCP scenario
data_8_5 = data.copy()
data_8_5['TS'] += temp_increase_8_5
data_8_5['RH2M'] += humidity_increase_8_5

data_4_5 = data.copy()
data_4_5['TS'] += temp_increase_4_5
data_4_5['RH2M'] += humidity_increase_4_5

data_2_6 = data.copy()  #No changes for RCP2.6

#RHcrit based on original formula
def compute_rh_crit(T):
    T = np.array(T)
    rh_crit = np.zeros_like(T)
    rh_crit[T <= 20] = -0.0026 * T[T <= 20]**3 + 0.160 * T[T <= 20]**2 - 3.13 * T[T <= 20] + 100.0
    rh_crit[T > 20] = 80
    return rh_crit

#Test the function with a range of temperatures
test_temperatures = np.linspace(0, 30, 100)
rh_crit_values = compute_rh_crit(test_temperatures)

plt.figure()
plt.plot(test_temperatures, rh_crit_values)
plt.xlabel('Temperature (°C)')
plt.ylabel('RH_crit (%)')
plt.title('Critical Relative Humidity as a function of Temperature')
plt.grid(True)
plt.show()

#Parameters
k11 = 0.578
k12 = 0.386
A = 0.3
B = 6
C = 1
p_T = 0.34 * 2
p_RH = 6.95 * 2
p_C = 33.01 * 2
W = 1
SQ = 1
C_decline = 1

#Mold growth index given a dataframe
def compute_mould_index(df):
    df['RHcrit'] = compute_rh_crit(df['TS'])

    #Initialize arrays for dM/dt and M
    dMdt = np.zeros(len(df))
    M = np.zeros(len(df))
    t_1 = None

    #Iterate over the data to compute dM/dt and M
    for i in range(len(df)):
        M_max = A + B * (df.loc[i, 'RHcrit'] - df.loc[i, 'RH2M']) / (df.loc[i, 'RHcrit'] - 100) - C * ((df.loc[i, 'RHcrit'] - df.loc[i, 'RH2M']) / (df.loc[i, 'RHcrit'] - 100))**2
        k1 = k11 if M[i-1] < 1 else k12
        
        #Calculate k2 with overflow handling
        if i > 0:
            exponent = 2.3 * (M[i-1] - M_max)
            if exponent > 700:
                k2 = 0
            else:
                k2 = max(1 - np.exp(exponent), 0)
        else:
            k2 = 1  #Default value for k2 when i is 0
            
        if df.loc[i, 'RH2M'] < df.loc[i, 'RHcrit'] and t_1 is None:
            t_1 = i
            
        elapsed_time = (i - t_1) * 24 if t_1 is not None else 0
        
        if df.loc[i, 'RH2M'] < df.loc[i, 'RHcrit']:
            if elapsed_time <= 6:
                dMdt[i] = -0.032 * C_decline
            elif elapsed_time <= 24:
                dMdt[i] = 0
            elif elapsed_time > 24:
                dMdt[i] = -0.0016 * C_decline
        else:
            dMdt[i] = (k1 * k2 / 7) * np.exp(-p_T * np.log(df.loc[i, 'TS']) - p_RH * np.log(df.loc[i, 'RH2M']) + 0.14 * W - 0.33 * SQ + p_C)
            t_1 = None
            
        M[i] = M[i-1] + dMdt[i] if i > 0 else 0

    df['dMdt'] = dMdt
    df['M'] = M

    return df

#Stochastic optimization for all RCP scenarios
num_variations = 50
results_8_5 = []
results_4_5 = []
results_2_6 = []

for _ in range(num_variations):
    # RCP8.5
    variation_df = data_8_5.copy()
    variation_df['TS'] += np.random.uniform(-0.5, 0.5)
    variation_df['RH2M'] += np.random.uniform(-5, 5)
    results_8_5.append(compute_mould_index(variation_df))

    # RCP4.5
    variation_df = data_4_5.copy()
    variation_df['TS'] += np.random.uniform(-0.5, 0.5)
    variation_df['RH2M'] += np.random.uniform(-5, 5)
    results_4_5.append(compute_mould_index(variation_df))

    # RCP2.6
    variation_df = data_2_6.copy()
    variation_df['TS'] += np.random.uniform(-0.5, 0.5)
    variation_df['RH2M'] += np.random.uniform(-5, 5)
    results_2_6.append(compute_mould_index(variation_df))

#Plot
plt.figure(figsize=(12, 7))
plt.hist([df['M'].iloc[-1] for df in results_2_6], bins=20, edgecolor='black', alpha=0.6, label='RCP2.6')
plt.hist([df['M'].iloc[-1] for df in results_4_5], bins=20, edgecolor='black', alpha=0.6, label='RCP4.5')
plt.hist([df['M'].iloc[-1] for df in results_8_5], bins=20, edgecolor='black', alpha=0.6, label='RCP8.5')
plt.title('Distribution of Mold Growth Index under Different RCP Scenarios')
plt.xlabel('Mold Growth Index (M)')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.show()

# CSV SAVE
def save_results_to_csv(results, scenario_name):

    combined_df = pd.concat([df[['Date', 'TS', 'RH2M', 'dMdt', 'M']] for df in results], ignore_index=True)
    output_file_path = f'results_{scenario_name}.csv'
    combined_df.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")

# Save results for each RCP scenario
save_results_to_csv(results_8_5, 'RCP8_5')
save_results_to_csv(results_4_5, 'RCP4_5')
save_results_to_csv(results_2_6, 'RCP2_6')
