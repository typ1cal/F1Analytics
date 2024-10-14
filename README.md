
# Formula 1 Data Analysis with Azure

This project analyzes Formula 1 race data to gain insights into team and driver performance, and develop predictive models for future races. It leverages various Azure services like Azure Blob Storage, Azure Databricks, and Azure Machine Learning to process and analyze the data.
![360_F_982018859_Otp0tiymYyGn2xNizt84c6UGXCMsxOY3](https://github.com/user-attachments/assets/e265e38d-b6fb-4fa5-9706-166fb4cb7d31)


## Project Goals

The goals of this project are as follows:
- Collect and preprocess data from the **Kaggle Formula 1 dataset**.
- Store the processed data in **Azure Blob Storage**.
- Analyze and visualize data using **Azure Databricks**.
- Develop predictive models for race results using **Azure Machine Learning**.

## Data Sources

- **Race results**: Includes data like finishing positions, lap times, and pit stops.
- **Driver standings**: Tracks points, wins, and podium finishes of drivers.
- **Team performance**: Focuses on team standings, wins, and championship results.

## Tools and Technologies

- **Azure Blob Storage**: Used for data storage.
- **Azure Databricks**: Employed for data preprocessing, analysis, and visualization.
- **Azure Machine Learning**: Used to build and train machine learning models.
- **Pyspark**: Used for processing large datasets within Databricks.

## Key Data Files

- `GrandPrix_drivers_details_1950_to_2022.csv`: Details of drivers from 1950 to 2022.
- `GrandPrix_fastest-laps_details_1950_to_2022.csv`: Data on fastest laps from 1950 to 2022.
- `GrandPrix_races_details_1950_to_2022.csv`: Race results data from 1950 to 2022.

## Steps Involved

1. **Data Collection**: Data from the Kaggle Formula 1 dataset is uploaded to Azure Blob Storage.
2. **Data Preprocessing**: Using Azure Databricks, the data is cleaned, transformed, and converted into Spark DataFrames for efficient processing.
3. **Data Analysis and Visualization**: The data is analyzed to generate insights on race results, driver performance, and team standings.
4. **Predictive Modeling**: Machine learning models are built and trained using Azure Machine Learning to predict future race outcomes.

## Visualizations

Here are a few standout visualizations from the analysis:

### 1. Driver Performance Over Time
A line graph showing how driver standings (points) change over time across multiple seasons. This helps visualize which drivers have consistently performed well or improved over the years.
![newplot](https://github.com/user-attachments/assets/f0c26299-c15c-40b3-a47a-5f4467a38af3)


### 2. Team Championships Standings
A bar chart comparing the championship standings of various teams across seasons. This helps identify dominant teams and their performance trends over the years.
![image](https://github.com/user-attachments/assets/229c29f5-987e-42bb-a719-aa77b244da7d)


### 3. Average Lap Analysis
A scatter plot visualizing the average laps achieved in various Grand Prix races, providing insights into track conditions, driver speed, and team performance.

![image](https://github.com/user-attachments/assets/9ada2ea3-f4bf-4776-95c7-7396063b9d4e)



## Project Structure

```
/F1DataAnalysis
|-- data/                         # Contains data files
|-- notebooks/                    # Jupyter notebooks used for data analysis
|-- README.md                     # Project documentation
```

## How to Run

1. **Setup Azure Services**: Ensure you have access to an Azure account with Blob Storage, Databricks, and PowerBI
2. **Upload Data**: Store the Formula 1 dataset in the appropriate Azure Blob Storage container.
3. **Run Notebooks**: Use the provided Jupyter notebooks to preprocess and analyze the data in Azure Databricks.
4. **Train Models**: Use the ML pipeline defined in the notebooks to train predictive models.

## Conclusion

This project provides deep insights into Formula 1 races and helps predict future race outcomes using a combination of Azure cloud services. It also showcases the power of integrating Azure for large-scale data processing and machine learning.

## Author

- Yash Wadhawe
- Final Project Assignment for IST615 - Cloud Management
