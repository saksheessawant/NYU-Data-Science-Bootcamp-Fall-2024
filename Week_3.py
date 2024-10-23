import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
url = "https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Task 1: Filter weekdays and plot pedestrian counts for each day of the week
df['Weekday'] = df['Date'].dt.weekday
weekdays_df = df[df['Weekday'] < 5]

# Group by weekday and get average pedestrian counts for each weekday
weekday_avg = weekdays_df.groupby('Weekday')['Pedestrians'].mean()

# Plotting weekday pedestrian counts
plt.figure(figsize=(10,6))
weekday_avg.plot(kind='line', marker='o', color='b')
plt.title('Average Pedestrian Counts for Each Weekday (Monday to Friday)')
plt.xlabel('Day of the Week (0=Monday, 4=Friday)')
plt.ylabel('Average Pedestrian Counts')
plt.grid(True)
plt.show()

# Task 2: Analyze pedestrian counts in 2019 and identify correlations with weather
df['Year'] = df['Date'].dt.year
df_2019 = df[df['Year'] == 2019]

# One-hot encode the 'Weather Summary' column
df_2019_encoded = pd.get_dummies(df_2019, columns=['Weather Summary'])

# Calculate correlation matrix for pedestrian counts and weather conditions
correlation_matrix = df_2019_encoded.corr()

# Display the correlation matrix for 'Pedestrians' and weather conditions
print(correlation_matrix['Pedestrians'].filter(like='Weather Summary'))

# Optional: Plot the correlation matrix heatmap
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix: Pedestrians and Weather Conditions (2019)')
plt.show()

# Task 3: Categorize time of day and analyze pedestrian activity patterns
def categorize_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

# Create a new column for the hour and time of day categories
df['Hour'] = df['Date'].dt.hour
df['Time of Day'] = df['Hour'].apply(categorize_time_of_day)

# Group by time of day and get average pedestrian counts
time_of_day_avg = df.groupby('Time of Day')['Pedestrians'].mean()

# Plotting pedestrian counts by time of day
plt.figure(figsize=(10,6))
time_of_day_avg.plot(kind='bar', color='orange')
plt.title('Average Pedestrian Counts by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Average Pedestrian Counts')
plt.grid(True)
plt.show()
