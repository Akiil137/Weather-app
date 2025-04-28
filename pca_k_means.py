import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from Weather_app import hashingit
from datetime import datetime
import re
import matplotlib.patches as patches
import reverse_geocode



def reset_global_variables():
    global superdf, allnames, i
    superdf = pd.DataFrame()
    allnames = pd.Series()
    i = 0

def find_outer_points(cluster_data):
    # Find the point with the smallest and largest y coordinates for each x coordinate
    max_y_points = cluster_data.loc[cluster_data.groupby('Longitude')['Latitude'].idxmax()]
    min_y_points = cluster_data.loc[cluster_data.groupby('Longitude')['Latitude'].idxmin()]

    # Find the point with the smallest and largest x coordinates for each y coordinate
    max_x_points = cluster_data.loc[cluster_data.groupby('Latitude')['Longitude'].idxmax()]
    min_x_points = cluster_data.loc[cluster_data.groupby('Latitude')['Longitude'].idxmin()]

    # Concatenate the results to get all outer points
    outer_points = pd.concat([max_y_points, min_y_points, max_x_points, min_x_points]).drop_duplicates()

    return outer_points

superdf = pd.DataFrame()
allnames = pd.Series()
i=0

# sanitizes name so that the directory will work
def sanitize_filename(filename):
    # Remove any characters that are not alphanumeric or underscore
    return re.sub(r'\W', '', filename)


# this is not only the pca function but also an intermediary where all the k- means and visuals are processed
# it will start by producing all the coordinats representing our rows respectively
def pca_(data, file_names, form_dict,country):
    
    form_hash = hashingit.hash_it_u(form_dict)
    
    # Split the elements into latitude and longitude
    latitude_longitude = file_names.str.split('_', expand=True)
    
    # dataFrame with separate columns for latitude and longitude
    xy = pd.DataFrame({
        'Latitude': latitude_longitude[0].astype(float),
        'Longitude': latitude_longitude[1].str.replace('.csv', '').astype(float)
    })

    print('pca step 1')

    scaler = StandardScaler()
    std_df = scaler.fit_transform(data)
    pca = PCA()
    print(std_df.shape)
    print(std_df.dtype)
    print('here')
    print(std_df)
    print('here')
    pca.fit(std_df)
    print(data.shape)
    cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
    component_number = 1

    for i in range(0,len(cumulative_variance_ratio)):
        if cumulative_variance_ratio[i] > 0.85:
            component_number = i
            print(f"this will require {component_number} principle components for accuracy")
            break


    pca = PCA(n_components=component_number)
    
    pca.fit(std_df)
    
    scores = pca.transform(std_df)
    clusters = k_means_inertia(scores)
    
    kmeans_pca = k_means(scores,clusters)
    
    std_df = pd.DataFrame(std_df)

    dspk = pd.concat([std_df,pd.DataFrame(scores)],axis=1)

    
    component_strings = [f'{len(std_df.columns)+i}' for i in range(1, component_number + 1)]
    
    dspk.columns.values[-component_number:]=component_strings
    dspk = pd.concat([dspk,xy],axis=1)
    print('pca part 2 step 4')
    print(len(std_df))
    
# =============================================================================
    # Compute silhouette score
    silhouette_avg = silhouette_score(scores, kmeans_pca.labels_)
    
    # Compute Davies-Bouldin index
    db_index = davies_bouldin_score(scores, kmeans_pca.labels_)
    
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {db_index}")
#     
# =============================================================================

    dspk['segment kmeans pca'] = kmeans_pca.labels_
    cluster_mapping = {i: f'Segment {i+1}' for i in range(clusters)}
    
    dspk['Segment'] = dspk['segment kmeans pca'].map(cluster_mapping)
   
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#')
    print('ltitude and longitude')
    print(xy.shape)
    print('dspk')
    print(dspk.shape)
    print('data')
    print(data.shape)
    print('standardised data')
    print(std_df.shape)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    if form_dict['cluster_by_country']:
        longitude_column = dspk.iloc[:, -3].values    
        latitude_column = dspk.iloc[:, -4].values
        print(latitude_column)
        print(longitude_column)
        print('just saying we got here with the coordinates')
        plt.figure()  # Create a new figure
        
        # Defines color palette
        custom_palette = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'black', 'gray', 'darkorange', 'maroon', 'deeppink', 'dimgray', 'olive', 'teal']
        
        # Creates a scatter plot for the data points
        sns.scatterplot(x=longitude_column, y=latitude_column, hue=dspk['Segment'], palette=custom_palette)
        # Find the outer data points for each cluster
        for i, cluster_label in enumerate(dspk['Segment'].unique()):
            cluster_data = dspk[dspk['Segment'] == cluster_label][['Longitude', 'Latitude']]
            outer_points = find_outer_points(cluster_data)
            
            # Create a polygon from the outer points and draw it around the cluster
            polygon = patches.Polygon(outer_points, edgecolor=custom_palette[i], linewidth=2, facecolor='none')
            plt.gca().add_patch(polygon)
        print('polygon clusters have just been made should be on it now')
        
        directory = f'Weather_app/static/images/unsupervised/cluster_by_country/{form_hash}'
        
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists before specifying file_path
        print('just made a new directory ')
        print(directory)
        s_country = sanitize_filename(country)
        image = f'{s_country}_image.png'
        print(f'{image}')
        file_path = os.path.join(directory, image)
        plt.title(s_country)
        
        # Place the legend outside the graph
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Now, create the scatter plot with outer shapes and save the figure
        plt.savefig(file_path, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend in the saved image
        print('image has been saved')
        return directory

        
    if form_dict['cluster_by_weather']:
        # Assuming that dspk already has the latitude and longitude columns
        latitude_column = dspk.iloc[:, -4].values
        longitude_column = dspk.iloc[:, -3].values
        
        # Initialize an empty list to store coordinates
        coordinates = []
        
        # Loop through the length of latitude_column (or longitude_column, they should have the same length)
        for i in range(len(latitude_column)):
            lat = latitude_column[i]
            lon = longitude_column[i]
            coordinates.append((lat, lon))
        
        # Use reverse_geocode.search for all coordinates at once
        all_data = reverse_geocode.search(coordinates)
        
        # Extract all countries from the data
        all_countries = [data['country'] for data in all_data]
        
        # Add 'Country' column to dspk
        dspk['Country'] = all_countries
    
        # Group by 'Country' and plot for each country
        for country, country_data in dspk.groupby('Country'):
            plt.figure()  # Create a new figure
            
            # Defines color palette
            custom_palette = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'black', 'gray', 'darkorange', 'maroon', 'deeppink', 'dimgray', 'olive', 'teal']
            
            # Creates a scatter plot for the data points
            sns.scatterplot(x=country_data['Longitude'], y=country_data['Latitude'], hue=country_data['Segment'], palette=custom_palette)
            
            # Find the outer data points for each cluster
            for i, cluster_label in enumerate(country_data['Segment'].unique()):
                cluster_data = country_data[country_data['Segment'] == cluster_label][['Longitude', 'Latitude']]
                outer_points = find_outer_points(cluster_data)
                
                # Create a polygon from the outer points and draw it around the cluster
                polygon = patches.Polygon(outer_points, edgecolor=custom_palette[i], linewidth=2, facecolor='none')
                plt.gca().add_patch(polygon)
            
            directory = f'Weather_app/static/images/unsupervised/cluster_by_weather/{form_hash}'
            os.makedirs(directory, exist_ok=True)  # Ensure the directory exists before specifying file_path
            s_country = sanitize_filename(country)
            image = f'{s_country}_image.png'
            file_path = os.path.join(directory, image)
            plt.title(s_country)
            
            # Place the legend outside the graph
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            
            # Now, create the scatter plot with outer shapes and save the figure
            plt.savefig(file_path, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend in the saved image
    
        return directory

    print('weather')
    print(form_dict['cluster_by_weather'])
    print('country')
    print(form_dict['cluster_by_country'])
    return directory



# this run k-means clustering on the program however it also will work for if there is only 1 cluster in that case
def k_means(scores,clusters):
    print('kmeans step 3')

    if clusters < 2:
        kmeans_pca = KMeans(n_clusters=1, random_state=57)
        return kmeans_pca.fit(scores)
    else:
        kmeans_pca = KMeans(n_clusters=clusters, init='k-means++', random_state=57)
        return kmeans_pca.fit(scores)
        
# this will validate that we have chosen the best value to represent the clustering process in our program
# it will look for the value which represent the elbow point where it is best representing of the clusters in our graph
def k_means_inertia(scores):
    print('inertia step 2')
    wcss = []
    if scores.shape[0] < 2:
        return 1
    for i in range(1, 16):
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=57)
        kmeans_pca.fit(scores)
        wcss.append(kmeans_pca.inertia_)

    # Normalize WCSS values between 0 and 1
    max_wcss = max(wcss)
    normalized_wcss = [j / max_wcss for j in wcss]

    # Identify the elbow point
    elbow_point = None
    prev_diff = 2
    for i, value in enumerate(normalized_wcss):
        diff = prev_diff - value
        if diff < 0.04:  # threshold can be adjustyed as needed
            elbow_point = i + 1
            break
        prev_diff = value

    if elbow_point is not None:
        print(f'Elbow Point (Optimal k): {elbow_point}')
        return elbow_point
    else:
        print('No suitable elbow point found.')
        return None

        
# this will remove columns where the value is none 
#
# we wont consider them when we run  the program for it 
# as per request of the form we will also select a period between a year for the program to run in the dataframe 
# this will be applied to every dataframe to be considered so thatthey will be able to be applied to on another
def process(dataframe, start_date, end_date, form_dict):

    variable_names = ['datetime','temp', 'tempmin', 'tempmax', 'humidity', 'precip', 'precipprob', 'precipcover', 'snowdepth', 'windspeed', 'cloudcover', 'solarenergy', 'uvindex']

    for attribute in variable_names:
        if attribute != 'datetime' and attribute in form_dict and form_dict[attribute] is None:
            dataframe = dataframe.drop(columns=attribute)   
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])  # Convert to datetime

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

        # Concatenate both dataframefs and sort based on the 'date' column
        dataframe = pd.concat([dataframe_start, dataframe_end]).sort_values(by='datetime')

    else:
        end_month_day = pd.to_datetime(end_date).strftime('%m-%d')
        # Include all values for specified months and days for the same year
        dataframe = dataframe[dataframe['datetime'].dt.strftime('%m-%d').between(start_month_day, end_month_day)].sort_values(by='datetime')
        dataframe.head()
    # Group by month and day, and calculate the mean for each group


    # Return the modified DataFrame
    return dataframe





# will get a data file and flatten it into a single row
def flatten(df,name,form_dict):
    # Read the selected file
    df.drop({'datetime','conditions'},axis=1,inplace=True)
    # Flatten the DataFrame into a single row with row numbers
    ndf = df.unstack().to_frame().T
    
    ndf.columns = ndf.columns.map('{0[0]}_{0[1]}'.format)     
    ndf['coord'] = name

    return ndf

# this function is more of an intermediary it essentially acquires the country we will work on 
#
# it takes the input the country folder then goes through all the files applying procceses which will
# return another dataframe then flatten which will turn the dataframe into a single row 
# (more detail in the function) it will then put that row into a larger dataframe
# to convert names into the dataframe to make sure the dataframe can work no numerical values can be used
# so the coord column is made into a index to then check for if the form asked for by weather or by country,
# by country will run the program
# conventionally running pca for each country with weather bool set totrue will run the program via
# weather where further process will be done
def process_directory(directory_path,country,form_dict,countries):
    print(country)

    global superdf
    global allnames
    global i
    directory = None
    # Create an empty DataFrame to store flattened rows
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    ###########################os.makedirs(output_path, exist_ok=True)
    # Pick a specific file in the directory

    # Iterate through files in the directory
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        print('processing '+file_name)
        # Check if the path is a file (not a subdirectory)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            #print(df.columns)
            # Reduce number of rows and remover uneeded columns
            df = process(df,form_dict['fromdate'],form_dict['todate'],form_dict)
            
            # Flatten and attach each file
            flattened_row = flatten(df,file_name,form_dict)

            # Append the flattened row to the result DataFrame
            df1 = pd.concat([df1, flattened_row], axis=0,ignore_index=True)

    file_names = df1['coord']
            
    #file = (country+"_cluster.csv")
    #output_file_path = os.path.join(output_path, file)
    #df1.to_csv(output_file_path,index=False)
        
    df1.set_index('coord',inplace=True)
    

    if form_dict['cluster_by_weather']:
        superdf = pd.concat([superdf, df1], axis=0, ignore_index=True)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(file_names)
        allnames = pd.concat([allnames, file_names], ignore_index=True)
        i += 1

        if i == len(countries):
                        
            # This will run on all countries' dataframe and should be processed to output an image of all countries
            directory = pca_(superdf, allnames, form_dict, country)
            print('it was cluster by weather')
            return directory

    elif form_dict['cluster_by_country']:
        # This will run on a single country's dataframe and should output an image of the country for all countries
        directory = pca_(df1, file_names, form_dict, country)
        i += 1
        if i == len(countries):
            print('it was cluster by country')
            return directory

    return directory



#==============================================================================
#==============================================================================
#==============================================================================


#this will go through each inputted country and apply a process onto it given that
#the data exists inside of it
def cluster(form_dict):
    reset_global_variables()
    print('here starts pca-kmeans')
    data_path = "Weather_app\cleaned_data"
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(os.getcwd())
    print('heres the dictionary')
    for key, value in form_dict.items():
        print(f"Key: {key}, Value: {value}")
        print(type(f'{value}'))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    variables_string = form_dict['location']
    countries = variables_string.split(',')
    
    # Remove the last empty string (resulting from the trailing comma)
    if countries[-1] == '':
        countries.pop()
    
    print(sorted(countries))
    
    directory = ''
    # Iterate through country folders in raw_data
    for country_folder in countries: #os.listdir(data_path):
        print('this is the country of something just saying its here')
        print(country_folder)
        
        input_country_path = os.path.join(data_path, country_folder)
        
        # Check if the path is a directory
        if os.path.isdir(input_country_path):

            directory = process_directory(input_country_path,country_folder,form_dict,countries)  
        else: 
            print('folder not availiable')
    return directory
