import pandas as pd
import numpy as np

# Set seed for identical generation
np.random.seed(42)
n_rows = 10000

print("Generating Physics-Informed Synthetic Industrial Data...")

# 1. Generate realistic values based on real industry data distributions
air_temp_k = np.random.normal(loc=300.0, scale=2.0, size=n_rows)  # Ambient temp in Kelvin
# Physics rule: Process temperature rises based on ambient temperature + friction heat
process_temp_k = air_temp_k + np.random.normal(loc=10.0, scale=1.0, size=n_rows)

rotational_speed_rpm = np.random.normal(loc=1500, scale=150, size=n_rows)
torque_nm = np.random.normal(loc=40, scale=10, size=n_rows)
tool_wear_mins = np.random.uniform(low=0, high=240, size=n_rows)

# 2. Target vector
failure = np.zeros(n_rows, dtype=int)

# 3. Inject Real Industrial Failure Modes (Physics-Informed)
for i in range(n_rows):
    # Failure Mode 1: Heat Dissipation Failure (Actual industry condition)
    temp_diff = process_temp_k[i] - air_temp_k[i]
    if temp_diff < 8.6 and rotational_speed_rpm[i] < 1380:
        failure[i] = 1

    # Failure Mode 2: Tool Wear Failure
    if tool_wear_mins[i] > 220 and torque_nm[i] > 55:
        failure[i] = 1

# 4. Construct DataFrame
df = pd.DataFrame({
    'Machine_ID': [f"MAC_{id_num:05d}" for id_num in range(1, n_rows + 1)],
    'Air_Temperature_K': np.round(air_temp_k, 1),
    'Process_Temperature_K': np.round(process_temp_k, 1),
    'Rotational_Speed_RPM': np.round(rotational_speed_rpm, 0).astype(int),
    'Torque_Nm': np.round(torque_nm, 1),
    'Tool_Wear_Mins': np.round(tool_wear_mins, 0).astype(int),
    'Failure': failure
})

df.to_csv("industrial_telemetry.csv", index=False)
print(f"Success! Generated {n_rows} rows based on real industrial distributions.")
print(f"Total realistic failures captured: {df['Failure'].sum()} instances.")