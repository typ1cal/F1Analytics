# Databricks notebook source
# MAGIC %md
# MAGIC # Formula 1 Data Analysis with Azure by Yash Wadhawe
# MAGIC 
# MAGIC This project focuses on analyzing Formula 1 race data to gain insights into team and driver performance and develop predictive models for future races. I am using Azure services such as Azure Blob Storage, Azure Databricks, and Azure Machine Learning to collect, store, analyze, and visualize data from the Kaggle Formula 1 dataset.
# MAGIC This project is a part of Final Project Assignment for the course IST615 - Cloud Management
# MAGIC ## Project Goals
# MAGIC 
# MAGIC - Collect and preprocess data from the Kaggle Formula 1 dataset
# MAGIC - Store data in Azure Blob Storage
# MAGIC - Use Azure Databricks to analyze and visualize data
# MAGIC - Develop predictive models using Azure Machine Learning
# MAGIC 
# MAGIC ## Scope
# MAGIC 
# MAGIC The project will focus on the following aspects of Formula 1 race data:
# MAGIC 
# MAGIC - Race results: including finishing positions, lap times, and pit stops
# MAGIC - Driver standings: including points, wins, and podium finishes
# MAGIC - Team performance: including points, wins, and championship standings

# COMMAND ----------

# Setting up the storage account key and name from Azure
storage_account_name = "ykwadhawproject"
storage_account_key = "8nTbDiIk43jtoSzbAG3FxbZ17rwmMGihsBhLnZG2huMH/qOc0AHB67Tp+gh/zpoUIYWIXfgEo4jX+ASt5P2vTg=="

# COMMAND ----------

# Configuring the Spark
spark.conf.set(
    "fs.azure.account.key.{}.blob.core.windows.net".format(storage_account_name),
    storage_account_key
)

# COMMAND ----------

# Checking if the files are present in our blob
container_name = "f1datafiles"
folder_name = "f1data"  # Remove this line if there is no folder
files = dbutils.fs.ls("wasbs://{}@{}.blob.core.windows.net/{}".format(container_name, storage_account_name, folder_name))

for file in files:
    print(file.name)



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Data Manipulation

# COMMAND ----------

# Loading the csv files into spark dataframe

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

driver_details_path = "wasbs://{}@{}.blob.core.windows.net/{}/GrandPrix_drivers_details_1950_to_2022.csv".format(container_name, storage_account_name, folder_name)
fastest_laps_path = "wasbs://{}@{}.blob.core.windows.net/{}/GrandPrix_fastest-laps_details_1950_to_2022.csv".format(container_name, storage_account_name, folder_name)
races_details_path = "wasbs://{}@{}.blob.core.windows.net/{}/GrandPrix_races_details_1950_to_2022.csv".format(container_name, storage_account_name, folder_name)

driver_details_spark_df = spark.read.csv(driver_details_path, header=True, inferSchema=True)
fastest_laps_spark_df = spark.read.csv(fastest_laps_path, header=True, inferSchema=True)
races_details_spark_df = spark.read.csv(races_details_path, header=True, inferSchema=True)

# COMMAND ----------

# Converting the spark dataframes to pandas dataframe
driver_details_df = driver_details_spark_df.toPandas()
fastest_laps_df = fastest_laps_spark_df.toPandas()
races_details_df = races_details_spark_df.toPandas()


# COMMAND ----------

display(driver_details_df.head())
display(fastest_laps_df.head())
display(races_details_df.head())


# COMMAND ----------

# There appears to be an error where the metadata for drivers and initials is exchanged, lets change that
# Assuming the incorrect column names are 'initials' for drivers and 'drivers' for initials
# races_details_df.rename(columns={"Initials": "temp", "Driver": "Initials"}, inplace=True)
# races_details_df.rename(columns={"temp": "Driver"}, inplace=True)
# Assuming the original column name is 'initials'
# races_details_df.rename(columns={"initials": "Initials"}, inplace=True)
# display(races_details_df.head())
# Display the race_details df to check if the metadata is now correct


# COMMAND ----------

# MAGIC %md
# MAGIC # N/A Values
# MAGIC One of the important aspects of data piplining is to identifying null values. This might result in bad data interpretation and might result to a wrong judgement. So it is important to identifying null values before working with data. Fortunately, we do not have any null values in our data.

# COMMAND ----------

display(driver_details_df.isnull().sum())
display(fastest_laps_df.isnull().sum())
display(races_details_df.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC # Top 20 Manufacturers by Points Ever

# COMMAND ----------

import matplotlib.pyplot as plt

# Calculate total points by constructors team
team_points = driver_details_df.groupby('Car')['PTS'].sum().reset_index()

# Sort by total points and select the top 20
team_points_sorted = team_points.sort_values('PTS', ascending=False).head(20)

# Line chart
plt.figure(figsize=(20, 8))
plt.plot(team_points_sorted['Car'], team_points_sorted['PTS'], marker='o')
plt.xlabel('Constructor', fontsize=14)
plt.ylabel('Total Points', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.title('Top 20 Manufacturers by Total Points', fontsize=16)
plt.tight_layout()
plt.grid()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Driver's Standing with Total Points Ever

# COMMAND ----------

# Group the driver_details_df DataFrame by 'Driver' and sum the 'PTS' column for each driver
driver_points_df = driver_details_df.groupby('Driver')['PTS'].sum().reset_index()

# Sort the resulting DataFrame by the 'PTS' column in descending order and select the top 10 rows
top_drivers_df = driver_points_df.sort_values('PTS', ascending=False).head(10)

# Print the top 10 drivers ever with highest points scored
print(top_drivers_df)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Analyze the distribution of lap times and how they have changed over the years
# MAGIC 
# MAGIC We'll try to identify lap times for every year separtely and then try to genenrate a graph for average lap times for every year.

# COMMAND ----------

import pandas as pd
# from summarytools import dfSummary
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
df_lap_times = races_details_df[['Laps', 'Year']]
df_lap_times['Laps'] = df_lap_times['Laps'] * 60
g = sns.FacetGrid(df_lap_times, col='Year', col_wrap=4, height=3, margin_titles=True)
g.map(plt.hist, 'Laps', bins=20)
avg_lap_times = df_lap_times.groupby('Year').mean()
plt.figure()
plt.plot(avg_lap_times.index, avg_lap_times['Laps'], marker='o')
plt.xlabel('Year')
plt.ylabel('Average Lap Time (seconds)')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Top Drivers by Points
# MAGIC 
# MAGIC We now try to develop a line graph for points to year graph for drivers. The longest line is of Sir Lewis Hamilton from 2014 to 2020 which shows a dominant 6 years for a F1 drivers, followed by Sebastian Vettel, and Fernando Alonso

# COMMAND ----------

df_driver_points = driver_details_df.groupby('Driver')['PTS'].sum().reset_index()
df_driver_points = df_driver_points.sort_values('PTS', ascending=False)
top_drivers = df_driver_points.head(10)
df_fastest_laps = fastest_laps_df
top_drivers_laps = pd.merge(top_drivers, df_fastest_laps, on='Driver')
fig = px.line(top_drivers_laps, x='Year', y='PTS', color='Driver', title='Top Drivers by Points')
fig.show()



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Barplots for Number of Grand Prix Races per Year
# MAGIC 
# MAGIC The graph shows a visual represents of the total number of grand prix races took place from 1950 to 2022. The number of races seem to increase but there is a little downfall in 2020, which can be because of Covid-19. A better idea can be seen when we hover the mouse around a bar plot down.

# COMMAND ----------

race_counts = races_details_df['Year'].value_counts()
fig = px.bar(x=race_counts.index, y=race_counts.values, labels={'x':'Year', 'y':'Number of Races'})
fig.update_layout(title='Number of Grand Prix Races per Year')
fig.show()

#display(driver_details_df.isnull().sum())
#display(fastest_laps_df.isnull().sum())
#display(races_details_df.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Most Successful Drivers and Teams in Terms of Race Wins
# MAGIC 
# MAGIC Displaying a beautiful graph that develops a scatter plot to identify best teams with their best drivers. The information in the visual graph is represented in a better sense when we hover the mouse on the scatter. The right most dot shows Lewis Hamilton with Mercedes has 82 wins as of 2022.

# COMMAND ----------

df_winners = races_details_df.groupby(['Driver', 'Car']).size().reset_index(name='Wins')
df_winners = df_winners.sort_values(by=['Wins'], ascending=False)
fig = px.scatter(df_winners, x="Wins", y="Driver", color="Car", size="Wins", hover_name="Car",
                 hover_data={"Wins": True},
                 title="Most Successful Drivers and Teams in Terms of Race Wins",
                 labels={"Wins": "Race Wins", "Winner": "Driver"})
fig.update_layout(xaxis_title="Race Wins", yaxis_title="Driver", legend_title="Team")
fig.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Number of Wins of a Driver with their Manufacturer
# MAGIC 
# MAGIC This graph provides an interesting bar graph of drivers with their total wins with the Manufacturer.

# COMMAND ----------

df_wins_per_car = races_details_df.groupby(['Car', 'Driver'])['Driver'].count().reset_index(name='Wins')
df_wins_per_car = df_wins_per_car.sort_values(by='Wins', ascending=False)
fig = px.bar(df_wins_per_car, x='Car', y='Wins', color='Driver')
fig.update_layout(title='Number of Wins per Car', xaxis_title='Car', yaxis_title='Number of Wins')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Driver's Year Wise Performance
# MAGIC 
# MAGIC This graph provides a interesting data of driver's year wise performance. Here Max almost has straight upward line from 2020 to 2022. Which explains the performance of Redbull Team.

# COMMAND ----------

df_driver = driver_details_df
df_year_driver_pts = driver_details_df.groupby(['Year', 'Driver']).sum()['PTS'].reset_index()
fig = px.line(df_year_driver_pts, x='Year', y='PTS', color='Driver', title='Year-wise Driver Performance')
fig.update_layout(xaxis_title='Year', yaxis_title='Points')
fig.show()
