import os
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

def merge_files_directory(path, master_df=None):
    if master_df is None:
        master_df = pd.DataFrame()

    csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
    for file in csv_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)

        file_prefix = file.split('.')[0][:3]
        print(file_prefix)
        df.columns = [f"{file_prefix}_{col}" if col != "Date" else col for col in df.columns]
        if master_df.empty:
            master_df = deepcopy(df)
        else:
            master_df = pd.merge(master_df, df, on='Date', how='outer')

    return master_df

def pre_process(master_df):
    #Dropping certain columns
    columns_to_drop = [col for col in master_df.columns if 'Vol' in col or 'Change' in col]
    master_df.drop(columns=columns_to_drop, inplace=True)

    #Sorting by date
    master_df['Date'] = pd.to_datetime(master_df['Date'], format='%m/%d/%Y')
    master_df = master_df[master_df['Date'].dt.year >= 2019]                    #Timeline change to assess effect
    master_df.sort_values(by='Date', ascending=True, inplace=True)
    master_df.reset_index(drop=True, inplace=True)

    #Formating columns to float
    print('Column dtypes:')
    print(master_df.dtypes)
    numeric_columns = master_df.select_dtypes(include=['object']).columns
    for col in numeric_columns:
        master_df.loc[:, col] = master_df[col].str.replace(",", "").astype(float)

    #Handling NaN values
    nan_count = master_df.isna().sum().sum()
    print('Number of NaNs -', nan_count)
    master_df = master_df.fillna(method='ffill')                                #Some variables miss values over weekend
    master_df = master_df.dropna()                                              #Starting rows, no previous val for ffill
    print('Number of NaNs -', master_df.isna().sum().sum())

    #Correlation matrix
    master_df_date_removed = deepcopy(master_df)
    master_df_date_removed.drop(columns='Date', inplace=True)
    correlation_matrix = master_df_date_removed.corr()
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

    #Keeping most interesting features after correlation analysis
    columns_to_drop = [col for col in master_df.columns if 'Date' not in col and 'Price' not in col
                       and 'BTC_Open' not in col and 'BTC_High' not in col and 'BTC_Low' not in col]
    #columns_to_drop = [col for col in master_df.columns if 'Date' not in col and 'BTC' not in col]   #Only BTC features
    master_df.drop(columns=columns_to_drop, inplace=True)

    #BTC price plot
    plt.plot(master_df.index, master_df['BTC_Price'])
    plt.xlabel('Days (Jan 01 2019 - Sep 15 2023)')
    plt.ylabel('BTC price in $')
    plt.title('BTC price plot')
    plt.grid(True)
    plt.show()

    return master_df


if __name__ == "__main__":
    raw_data_path = r'C:\Users\vijay\Desktop\Personal_projects\BTC-pred-model\raw_data' #Change accordingly
    data_path = r'C:\Users\vijay\Desktop\Personal_projects\BTC-pred-model'              #Change accordingly

    master_df = merge_files_directory(raw_data_path)
    print('Master df after merge:')
    print(master_df.shape)

    master_df_pre_process = pre_process(master_df)
    print('Master df after pre-processing:')
    print(master_df_pre_process.shape)

    merged_pre_processed_data_path = os.path.join(data_path, 'merged_preprocessed_data.csv')
    master_df_pre_process.to_csv(merged_pre_processed_data_path, index=False)

    





