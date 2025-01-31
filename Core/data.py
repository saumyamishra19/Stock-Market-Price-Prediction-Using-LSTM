import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler as MM


class Data :

    def __init__(self) :
        self.dataframe = pd.DataFrame([])

    #Read data from CSV dataset(s)
    def read(self, filename : str) :
        """Reads values/data from CSV files into dataframe"""
        if os.path.exists(filename) :
            self.dataframe = pd.read_csv(filename)
            print(f"Data successfully read from {filename}.")
        else :
            print(f"Error: The file {filename} does not exist.")

    #Check for null values in the dataset(s)
    def check_null_values(self) :
        """Checks for null values in the dataframe."""
        print("Null Values Before Cleaning:")
        print(self.dataframe.isnull().sum())

    #Cleaning the dataset :
    def clean_dataset(self) :
        """Cleans the dataset by removing null values"""
        if self.dataframe is not None :
            original_shape = self.dataframe.shape #store the original shape of the DataFrame
            self.dataframe = self.dataframe.dropna() #drop all rows with null values; dropna() is a Pandas method that, by default, drops all rows that contain at least one null value.
            cleaned_shape = self.dataframe.shape #store the cleaned shape of DataFrame
            print(f"Dataset cleaned. Original shape: {original_shape}, Cleaned shape: {cleaned_shape}.")
        else:
            print("Error: No data to clean. Please read a dataset first.")

    # Normalize the data
    def normalize(self):
        """Normalizes numeric columns in the dataframe"""
        # Select columns to scale
        columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        # Ensure all specified columns are in the DataFrame
        columns_to_scale = [col for col in columns_to_scale if col in self.dataframe.columns]

        if columns_to_scale:
            # Select the data to scale
            data_to_scale = self.dataframe[columns_to_scale]

            # Initialize the MinMaxScaler
            self.scaler = MM()

            # Fit and transform the data
            scaled_data = self.scaler.fit_transform(data_to_scale)

            # Update the DataFrame with the scaled data
            self.dataframe[columns_to_scale] = scaled_data

            print("Data successfully normalized:")
            print(self.dataframe.head())
        else:
            print("Error: No columns to normalize. Please ensure the dataframe contains the necessary columns.")

    def visualize_open(self, y_label):
        """Plots graph for open stock values vs. date."""
        if 'Date' in self.dataframe.columns and y_label in self.dataframe.columns:
            self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'])
            plt.figure(figsize=(14, 7))
            sns.lineplot(data=self.dataframe, x='Date', y=y_label)
            plt.title('Open Stock Values vs. Date')
            plt.xlabel('Date')
            plt.ylabel(f'{y_label} Stock Value')
            plt.show()
        else:
            print(f"Dataframe does not contain 'Date' and '{y_label}' columns.")

    def visualize_close(self, y_label):
        """Plots graph for open stock values vs. date."""
        if 'Date' in self.dataframe.columns and y_label in self.dataframe.columns:
            self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'])
            plt.figure(figsize=(14, 7))
            sns.lineplot(data=self.dataframe, x='Date', y=y_label)
            plt.title('Close Stock Values vs. Date')
            plt.xlabel('Date')
            plt.ylabel(f'{y_label} Stock Value')
            plt.show()
        else:
            print(f"Dataframe does not contain 'Date' and '{y_label}' columns.")


    def print_head(self):
        """Prints the head of the dataframe."""
        print("Head of the DataFrame:")
        print(self.dataframe.head())

    def print_description(self):
        """Prints the description of the dataframe."""
        print("\nDescription of the DataFrame:")
        print(self.dataframe.describe())
