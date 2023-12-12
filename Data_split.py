import pandas as pd
import re
from sklearn.model_selection import train_test_split

weather_csv_file_path = 'weather.csv'
energy_csv_file_path = 'energy.csv'  

weather_df = pd.read_csv(weather_csv_file_path)
energy_df = pd.read_csv(energy_csv_file_path)

selected_weather_df = weather_df.iloc[:, [0, 2, 6, 7]]
selected_energy_df = energy_df.iloc[:, [0, 21, 26]]

merged_df = pd.merge(selected_weather_df, selected_energy_df, left_on=selected_weather_df.columns[0], right_on=selected_energy_df.columns[0])
merged_df = merged_df.drop(columns=['time'])
merged_df['temp_celsius'] = (merged_df['temp'] - 273.15).round(2)
pattern = r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):\d{2}:\d{2}\+(\d{2})'
merged_df['Month'] = merged_df['dt_iso'].apply(lambda x: re.search(pattern, x).group(2))
merged_df['Hour'] = merged_df['dt_iso'].apply(lambda x: re.search(pattern, x).group(4))
merged_df = merged_df.drop(columns=['dt_iso'])
merged_df = merged_df.drop(columns=['temp'])
merged_df['Month'] = merged_df['Month'].astype('int64')
merged_df['Hour'] = merged_df['Hour'].astype('int64')

new_order = ['Month', 'Hour' , 'temp_celsius', 'humidity', 'wind_speed', 'generation wind onshore', 'total load actual']
merged_df = merged_df[new_order]

df_shuffled = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df, test_df = train_test_split(df_shuffled, test_size=0.2, random_state=42)

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)