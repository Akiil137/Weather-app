import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
from datetime import datetime
import matplotlib.patches as patches
import reverse_geocode
from Weather_app import hashingit
import matplotlib.pyplot as plt
import re
import math



superdf = pd.DataFrame()
allnames = pd.Series()
i=0

def reset_global_variables():
    global superdf, allnames, i
    superdf = pd.DataFrame()
    allnames = pd.Series()
    i = 0


import pandas as pd

def total_distance_of_line(outer_points):
    total_distance = 0
    
    for i in range(len(outer_points) - 1):
        point1 = outer_points.iloc[i]
        point2 = outer_points.iloc[i + 1]
        distance = math.sqrt((point2['Longitude'] - point1['Longitude'])**2 + (point2['Latitude'] - point1['Latitude'])**2)
        total_distance += distance
    
    return total_distance



def euclidean(x, y):
    # Calculate Euclidean distance between two points x and y
    return ((x['Longitude'] - y['Longitude'])**2 + (x['Latitude'] - y['Latitude'])**2) ** 0.5


def alarm(outer_points, threshold=3.0):
    for i in range(len(outer_points) - 1):
        point1 = outer_points.iloc[i]
        point2 = outer_points.iloc[i + 1]
        distance = euclidean(point1, point2)
        if distance > threshold:
            print(f"Alarm: Distance between points {i} and {i+1} is greater than {threshold}.")


def find_outer_points(cluster_data):
    # Your code for finding outer points here...

    # Reset the display settings after you've printed the DataFrame
    
    # Find the point with the smallest and largest y coordinates for each x coordinate
    max_y_points = cluster_data.loc[cluster_data.groupby('Longitude')['Latitude'].idxmax()]
    min_y_points = cluster_data.loc[cluster_data.groupby('Longitude')['Latitude'].idxmin()]

    # Find the point with the smallest and largest x coordinates for each y coordinate
    max_x_points = cluster_data.loc[cluster_data.groupby('Latitude')['Longitude'].idxmax()]
    min_x_points = cluster_data.loc[cluster_data.groupby('Latitude')['Longitude'].idxmin()]
    
    # Concatenate the results to get all outer points
    outer_points = pd.concat([max_y_points, min_y_points, max_x_points, min_x_points]).drop_duplicates().reset_index(drop=True)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Initial outer points:')
    print(outer_points)
    print(total_distance_of_line(outer_points))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    outer_points = reorder_outer_points(outer_points)
    
    print('Updated outer points:')
    print(outer_points)
    
    print(total_distance_of_line(outer_points))
    return outer_points




def reorder_outer_points(outer_points):
    # Implement your algorithm to reorder the outer points here...
    # You can use a heuristic algorithm like the nearest neighbor algorithm
    # to minimize the total distance of the line
    
    # Example:
    # Reorder the outer points using nearest neighbor algorithm
    reordered_points = [outer_points.iloc[0]]  # Start with the first point
    remaining_points = outer_points.drop(0)
    
    while len(remaining_points) > 0:
        last_point = reordered_points[-1]
        nearest_point_idx = remaining_points.apply(lambda row: euclidean(row, last_point), axis=1).idxmin()
        nearest_point = remaining_points.loc[nearest_point_idx]
        reordered_points.append(nearest_point)
        remaining_points = remaining_points.drop(nearest_point_idx)
    
    return pd.DataFrame(reordered_points)


# sanitizes name so that the directory will work
def sanitize_filename(filename):
    # Remove any characters that are not alphanumeric or underscore
    return re.sub(r'\W', '', filename)


def SVM_model(data, file_names, form_dict,country): 

    weather_attributes = ['temp', 'tempmin', 'tempmax', 'humidity', 'precip', 'precipprob',
                      'precipcover', 'snowdepth', 'windspeed', 'cloudcover', 'solarenergy', 'uvindex']

    # Filter form_dict to include only weather attributes
    weather_dict = {key: value for key, value in form_dict.items() if value is not None and key in weather_attributes}


    
    # Convert filtered dictionary to DataFrame row
    df_row = pd.DataFrame([weather_dict])

    
    # Create Hash for Supervised model
    form_hash = hashingit.hash_it_s(form_dict)
    
    
    # Split the elements into latitude and longitude
    latitude_longitude = file_names.str.split('_', expand=True)
    
    # Create a DataFrame with separate columns for latitude and longitude
    xy = pd.DataFrame({
        'Latitude': latitude_longitude[0].astype(float),
        'Longitude': latitude_longitude[1].str.replace('.csv', '').astype(float)
    })
    latitude_column = xy['Latitude']
    longitude_column = xy['Longitude']
    coordinates = []
    for i in range(len(latitude_column)):
        lat = latitude_column[i]
        lon = longitude_column[i]
        coordinates.append((lat, lon))    

    # Load the dataset
    # Use reverse_geocode.search for all coordinates at once
    all_data = reverse_geocode.search(coordinates)
    
    # Extract all countries from the data
    all_countries = [data['country'] for data in all_data]
    df = data


    # Separate features and target variable

    print('this is the df')
    print(df)
    print(df.columns)

    
    X = df.drop(columns=['datetime','snow','conditions','sunlight'])
    print('this is x')
    print(X)
    

    y = df['conditions']

    # Split the data into training and testing sets (70-30 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=92)
    
    # Initialize and train the Support Vector Classifier (SVM) model
    svm_model = SVC(kernel='rbf')  
    svm_model.fit(X_train, y_train)
    
    # Predict the labels for the test set
    y_pred = svm_model.predict(X_test)
    
    # Print the predicted labels
    condition = svm_model.predict(df_row)

    # Evaluate the model's accuracy


# =============================================================================
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of SVM model:", accuracy)
    
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
# =============================================================================

    print(condition)
    # dataframe representing for each row in the datframe the latitude longittude coordinates as well as the country where they are in
    df['Country'] = all_countries
    df['Latitude'] = latitude_column
    df['Longitude'] = longitude_column
    
    # Create Directory where the images will be stored based on the hash value from the input and format accordingly     
    directory = os.path.join(os.getcwd(), f'Weather_app/static/images/supervised/SVM/{form_hash}')
    
    # directory is now able to be used in webpage visualisations
    directory = directory.replace("\\", "/")
    os.makedirs(directory, exist_ok=True)
    print('just made a new directory ')
    print(directory)
    
    print('iconic')
    
    df_condition = df[df['conditions'] == condition[0]]
    
    print('}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}')
    print(df_condition)
    print('}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}')
    # Group by country and count the occurrences
    country_counts = df_condition.groupby('Country').size()
    
    ylim_min = 0  # Minimum y-axis limit
    ylim_max = country_counts.max() * 1.1  # Maximum y-axis limit
    
    # Plot the bar graph
    plt.figure(figsize=(12, 6))
    country_counts.plot(kind='bar')
    plt.title(f'Frequency of being {condition[0]} in Each Country')
    plt.xlabel('Country')
    plt.ylabel('Frequency')
    plt.ylim(ylim_min, ylim_max)  # Set y-axis limits
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    
    image = 'frquency_image.png'
    file_path = os.path.join(directory, image)
    print(country_counts)
    print('shoulda saved the frequency')
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()
    
    # Calculate the total number of rows for each country
    total_rows_by_country = df.groupby('Country').size()
    
    # Calculate the frequency of the predicted condition as a percentage
    condition_percentage = (country_counts / total_rows_by_country) * 100
    
    # Plot the bar graph
    plt.figure(figsize=(12, 6))
    condition_percentage.fillna(0).plot(kind='bar')  # Fill NaN values with 0
    plt.title('Percentage of Condition in Each Country')
    plt.xlabel('Country')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    
    image = 'percentage_match_image.png'
    file_path = os.path.join(directory, image)
    print('shoulda saved the percentage')
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()
    
    unique_countries = df['Country'].unique()
    
    # Plot scatter plots for each country
    for country in unique_countries:
        # Filter data for the current country
        country_df = df[df['Country'] == country]
    
        # Select rows with the same condition value as the predicted condition
        condition_df = country_df[country_df['conditions'] == condition[0]]
    
    
    
        # Group by latitude and longitude and count the occurrences of the condition
        coordinate_counts = condition_df.groupby(['Latitude', 'Longitude']).size().reset_index(name='Count')
        df2 = df.groupby(['Latitude', 'Longitude']).size().reset_index(name='max')
        coordinate_counts['max'] = df2['max']
    
        # Normalize the count to use it as the opacity value
        coordinate_counts['Opacity'] = coordinate_counts['Count'] / coordinate_counts['max']
        # Replace NaN values with 0
        coordinate_counts['Opacity'].fillna(0, inplace=True)
        
        # Plot scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(coordinate_counts['Longitude'], coordinate_counts['Latitude'],
                    s=100, alpha=coordinate_counts['Opacity'], edgecolor='black',color='grey')
        plt.title(f'Coordinates with Opacity based on Count of Condition in {country}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')


        legend_elements = []
        for opacity_level in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            legend_elements.append(patches.Circle((0, 0), 1, facecolor='grey', alpha=opacity_level))
        opacity_labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
        plt.legend(legend_elements, opacity_labels, loc='upper right', bbox_to_anchor=(1.05, 1), title='Opacity')



        cluster_data = country_df[['Longitude', 'Latitude']]
        # Find the outer points for the cluster
        outer_points = find_outer_points(cluster_data)
        
        # Create a polygon from the outer points and draw it around the cluster
        polygon = patches.Polygon(outer_points, edgecolor='red', linewidth=2, facecolor='none')
        plt.gca().add_patch(polygon)  # Add the polygon to the plot    


        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(cluster_data)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
        s_country = sanitize_filename(country)
        image = f'{s_country}_image.png'
        file_path = os.path.join(directory, image)
        plt.savefig(file_path, bbox_inches='tight')
        plt.show()
    
    if os.path.exists(directory):
        print("Directory exists:", directory)
    else:
        print("Directory does not exist:", directory)
    
    print(os.getcwd())
    print(directory)
    print('returned from the modeller')
    return directory
        


def process(dataframe, start_date, end_date, form_dict):
    # Names of all the Features in the data
    variable_names = ['datetime','temp', 'tempmin', 'tempmax', 'humidity', 'precip', 'precipprob', 'precipcover', 'snowdepth', 'windspeed', 'cloudcover', 'solarenergy', 'uvindex']

    # Loop Through each feature will select every feature which has a null value and remove from dataframe
    for attribute in variable_names:
        if attribute != 'datetime' and attribute in form_dict and form_dict[attribute] is None:
            dataframe = dataframe.drop(columns=attribute) 

    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])  # Convert to datetime
    

    # consider the date only by month and year (key for analysis of data)
    start_month_day = pd.to_datetime(start_date).strftime('%m-%d')
    
    # Include all values for specified months and days for the start_date's year
    start_date = datetime.strptime(str(start_date), '%Y-%m-%d')
    end_date = datetime.strptime(str(end_date), '%Y-%m-%d')
    # Check if end_date is in the next year

    
    if end_date.year > start_date.year:
        # Include all values from the start_date to the 31st of December of that year
        start_month_day = pd.to_datetime(start_date).strftime('%m-%d')
        dataframe_start = dataframe[dataframe['datetime'].dt.strftime('%m-%d').between(start_month_day, '12-31')]

        # Include all values from January 1st to the specified end_date for the same year
        end_month_day = pd.to_datetime(end_date).strftime('%m-%d')
        dataframe_end = dataframe[dataframe['datetime'].dt.strftime('%m-%d').between('01-01', end_month_day)]

        # Concatenate both dataframes and sort based on the 'date' column
        dataframe = pd.concat([dataframe_start, dataframe_end]).sort_values(by='datetime')

    else:
        end_month_day = pd.to_datetime(end_date).strftime('%m-%d')
        # Include all values for specified months and days for the same year
        dataframe = dataframe[dataframe['datetime'].dt.strftime('%m-%d').between(start_month_day, end_month_day)].sort_values(by='datetime')

    # Group by month and day, and calculate the mean for each group


    # Return the modified DataFrame
    # Data frame will now be a matrix wiht where the row s are only months and days between the time period selected
    return dataframe


def process_directory(directory_path,country,form_dict,countries):
    print('proccesing directory')
    print(country)

    global superdf
    global allnames
    global i
    directory = None
    # Create an empty DataFrame to store flattened rows
    df = pd.DataFrame()
    df1 = pd.DataFrame()

    # Pick a specific file in the directory

    # Iterate through files in the directory
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        print('processing '+file_name)
        # Check if the path is a file (not a subdirectory)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)

            # Reduce number of rows and remover uneeded columns
            df = process(df,form_dict['fromdate'],form_dict['todate'],form_dict)
            df['coord'] = file_name            

            # Append the flattened row to the result DataFrame
            df1 = pd.concat([df1, df], axis=0,ignore_index=True)

    file_names = df1['coord']
            
    df1.set_index('coord',inplace=True)
    
    # This will run on a single country's dataframe and should output an image of the country for all countries

    print('something should be between this')
    if form_dict['cluster_by_weather']:
        superdf = pd.concat([superdf, df1], axis=0, ignore_index=True)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(file_names)
        allnames = pd.concat([allnames, file_names], ignore_index=True)
        i += 1
        print('the value of i is this shoulsnt be any longer than the length')
        print(i<=len(countries))
        print(i)
        if i == len(countries):
                        
            # This will run on all countries' dataframe and should be processed to output an image of all countries
            directory = SVM_model(superdf, allnames, form_dict, country)
            print('it was cluster by weather')
            return directory
            
    if i == len(countries):    
        print('shouldnt see this but its ok')
        print(directory)
        print(form_dict['cluster_by_weather'])
        print('returned from the process directories')
        return directory



def predict(form_dict):
    reset_global_variables()
    form_dict.pop('csrf_token', None)
    desired_order = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'precipprob', 'precipcover', 
                 'snowdepth', 'windspeed', 'cloudcover', 'solarenergy', 'uvindex', 'fromdate', 
                 'todate', 'location', 'cluster_by_country', 'cluster_by_weather']
    
    # Create a new dictionary with keys ordered as desired
    ordered_form_dict = {key: form_dict[key] for key in desired_order if key in form_dict}
    print('starts svm modeller')

    data_path = "Weather_app\cleaned_data"
    print('correct order')
    
    for key, value in ordered_form_dict.items():
        print(f"Key: {key}, Value: {value}")
        print(type(value))

    variables_string = form_dict['location']
    countries = variables_string.split(',')
    
    # Remove the last empty string (resulting from the trailing comma)
    if countries[-1] == '':
        countries.pop()
    
    print(sorted(countries))
    
    directory = ''
    # Iterate through country folders in raw_data
    for country_folder in countries: #os.listdir(data_path):
        print(country_folder)
        
        input_country_path = os.path.join(os.getcwd(),data_path, country_folder)
        
        # Check if the path is a directory
        if os.path.isdir(input_country_path):

            directory = process_directory(input_country_path,country_folder,ordered_form_dict,countries)  
        else: 
            print(input_country_path)
            print('folder not availiable')

    return directory
