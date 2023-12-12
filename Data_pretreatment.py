import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

weather_csv_file_path = 'weather.csv'
energy_csv_file_path = 'energy.csv'  

weather_df = pd.read_csv(weather_csv_file_path)
energy_df = pd.read_csv(energy_csv_file_path)

selected_weather_df = weather_df.iloc[:, [0, 2, 6, 7]]
selected_energy_df = energy_df.iloc[:, [0, 21]]

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

new_order = ['Month', 'Hour' , 'temp_celsius', 'humidity', 'wind_speed', 'generation wind onshore']
merged_df = merged_df[new_order]

#print(weather_df.info())
#print(weather_df.head()) 

#print(energy_df.info())
#print(energy_df.head()) 

print(merged_df.info())
print(merged_df.head()) 



#월별 평균 풍력 발전량
monthly_avg_generation = merged_df.groupby('Month')['generation wind onshore'].mean()
print(monthly_avg_generation)
plt.figure(figsize=(10, 6))
monthly_avg_generation.plot(kind='bar', color='skyblue')
plt.title('Monthly Average Wind Generation')
plt.xlabel('Month')
plt.ylabel('Average Generation')
plt.xticks(rotation=45)
plt.show()


# 풍속에 따른 평균 발전량
grouped_data = merged_df.groupby(['wind_speed', 'Month'])['generation wind onshore'].agg(['mean', 'std']).reset_index()
filtered_data = grouped_data[grouped_data['wind_speed'] <= 10]
grouped_by_wind_speed = filtered_data.groupby('wind_speed')['mean'].mean().reset_index()
print(grouped_by_wind_speed)


plt.figure(figsize=(12, 6))
sns.barplot(x='wind_speed', y='mean', data=grouped_by_wind_speed, color='blue')
plt.title('Average Wind Generation by Wind Speed (up to 10)')
plt.xlabel('Wind Speed')
plt.ylabel('Average Generation')
plt.show()

#시간대별 습도 발전량 평균
grouped_by_hour = merged_df.groupby('Hour').agg({'generation wind onshore': 'mean', 'humidity': 'mean'})
print(grouped_by_hour)

grouped_by_hour['humidity_scaled'] = grouped_by_hour['humidity'] * (7000 - 4000) / 100 + 4000
plt.figure(figsize=(12, 6))

sns.lineplot(data=grouped_by_hour, x=grouped_by_hour.index, y='generation wind onshore', marker='o', label='Average Wind Generation')
sns.lineplot(data=grouped_by_hour, x=grouped_by_hour.index, y='humidity_scaled', marker='o', label='Scaled Average Humidity', color='green')

plt.title('Hourly Average Wind Generation and Scaled Humidity')
plt.xlabel('Hour')
plt.ylabel('Average Value / Scaled Value')
plt.legend()
plt.show()

#히트맵
plt.figure(figsize=(10, 8))
sns.heatmap(merged_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Variables")
plt.show()