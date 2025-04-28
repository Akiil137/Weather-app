# module imports
import os
import sys
import pandas as pd

'''
    This file is a a part of the data pipeline which will prepare data for
    transformation removing of errounous data and filtering outlier values
    the program will also properly fail if failure occurs 
'''


def process_country_folder(country,output):

    # Aquire the directories where we will get the data and send then clean the data
    input_country = os.path.join(os.getcwd(), 'Weather_app', 'raw_data')
    output_country = os.path.join(os.getcwd(), 'Weather_app', 'cleaned_data')
    os.makedirs(output_country, exist_ok=True)
    
    
    # Select the specific country folder in the directories to place data within
    input_country_path = os.path.join(input_country, country)
    output_country_path = os.path.join(output_country, country)
    
    # Create the corresponding processed country folder
    os.makedirs(output_country_path, exist_ok=True)

    # Iterate through files in the raw data country folder
    for file_name in os.listdir(input_country_path):
        input_file_path = os.path.join(input_country_path, file_name)

        # Check if the path is a file (not a subdirectory)
        if os.path.isfile(input_file_path):
            output_file_path = os.path.join(output_country_path, file_name)
            output.append('Cleaning coordinate data ' + str(file_name) +': '+ str(country) )
            
            # Apply cleaining transformations
            clean_file = clean(input_file_path,file_name)
            
            output.append('Cleaning Complete!')
            print('Cleaning coordinate data ' + str(file_name) +': '+ str(country) )
            contains_null_values = clean_file.isnull().any()
            
            # Checks to see if null values exist captures unexexcuted cleaning transformation
            # in case there are new errors
            if contains_null_values.any():
                # Identify the file where the new errornous data is and make new changes to clean function
                # to fix error
                print("File:"+file_name)
                print(contains_null_values[contains_null_values])
                print(contains_null_values[contains_null_values].info())
                sys.exit()
            
            # converts data/dataframe to csv and saves to clean data folder for its country
            clean_file.to_csv(output_file_path , index=False)
    return output                       
            
            
            

'''
    This function proccesses/cleans data based on
    . chosen attributes to removed based on PCA
    . outlier data found in each attibute (additional needs to be functionality can be added)
    . null values imputated or removed based conditions 

'''



def clean(input_dir,file_name): # this is the file of a country

    # Use the pandas library to make a dataframe 
    data = pd.read_csv(input_dir,parse_dates=['datetime','sunset','sunrise'])
    data['name'] = file_name
    for index in range(len(data)):
        row = data.iloc[index]
    
        # append conditions based on numerical values
        if row['cloudcover'] < 20 and row['solarenergy'] > 19:
            row['conditions'] += ", Sunny"
        if row['windspeed'] > 20:
            row['conditions'] += ", Windy"


    data.drop({'name','moonphase','feelslikemax','feelslikemin','dew','feelslike','preciptype','sealevelpressure','visibility','winddir','windgust','icon','stations','description','solarradiation','severerisk'},axis=1,inplace=True)

    for column in data.columns:
        selected_column = data[column]
        null_count = selected_column.isnull().sum()
        window_size = 7
        if selected_column.dtype == "float64":
            
            # Check if more than 80% of values are null
            if null_count / data.shape[0] > 0.8 :
                selected_column.fillna(0, inplace=True)

            # Iterate through the DataFrame
            for missing_point in data[selected_column.isnull()].index:
                # Get the previous and next 7 values with boundary checks
                start_index = max(0, missing_point - window_size)
                end_index = min(data.shape[0], missing_point + window_size + 1)

                previous_values = selected_column.iloc[start_index:missing_point]
                next_values = selected_column.iloc[missing_point + 1:end_index]

                # Concatenate the previous and next values
                surrounding_values = pd.concat([previous_values, next_values])

                # Fill missing value with the mean of surrounding values
                data.at[missing_point, column] = surrounding_values.mean()
                
                surrounding_null_indices = surrounding_values.index[surrounding_values.isnull()]
                data.loc[surrounding_null_indices, column] = 0                
                

        elif selected_column.dtype == "int64":
            if null_count / data.shape[0] > 0.8 :
                selected_column.fillna(0, inplace=True)            

            # Iterate through the DataFrame
            for missing_point in data[selected_column.isnull()].index:
                # Get the previous and next 7 values with boundary checks
                start_index = max(0, missing_point - window_size)
                end_index = min(data.shape[0], missing_point + window_size + 1)

                previous_values = selected_column.iloc[start_index:missing_point]
                next_values = selected_column.iloc[missing_point + 1:end_index]

                # Concatenate the previous and next values
                surrounding_values = pd.concat([previous_values, next_values])

                # Fill missing value with the mode of surrounding values
                data.at[missing_point, column] = surrounding_values.mode().iloc[0]
                
    data.drop(data.index[-1], inplace=True)
    data['sunlight'] = (data['sunset'] - data['sunrise']).dt.total_seconds() / 60
    data['sunlight'].fillna(method='ffill', inplace=True)
    data['sunlight'].fillna(method='bfill', inplace=True)
    
    data.drop({'sunset','sunrise'},axis=1,inplace=True)
    
    return data

#process_raw_data()







#this was to run on an entire folder of countries            
'''
def process_raw_data():
    raw_data_path = "raw_data"
    processed_data_path = "cleaned_data"
    # Create the processed_data folder
    os.makedirs(processed_data_path, exist_ok=True)

    # Iterate through country folders in raw_data
    for country_folder in os.listdir(raw_data_path):
        print(country_folder)
        input_country_path = os.path.join(raw_data_path, country_folder)
        output_country_path = os.path.join(processed_data_path, country_folder)

        # Check if the path is a directory
        if os.path.isdir(input_country_path):
            process_country_folder(input_country_path, output_country_path)

# Execute the data processing system
'''