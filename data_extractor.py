import os
import csv
import urllib.request
import codecs
from global_land_mask import globe
import reverse_geocode

'''
This function will aquire a country weather data based on the time period (fromdate,todate)
and resolution the user wants the data to be outputted
'''

def Extract(lat_res,lon_res,picked_country,fromdate,todate):
    try: 
        #path = os.join = os.path.join(os.getcwd(), 'Weather_app')
        #weather_data_path = os.path.join(path, 'raw_data')
        
        # Select the directory area we want to save the data
        weather_data_path = os.path.join(os.getcwd(), 'Weather_app', 'raw_data')
        weather_data_pathc = os.path.join(os.getcwd(), 'Weather_app', 'cleaned_data')


        output = []
        # Select the folder where we are going to save the data for organisation
        directory_path = os.path.join(weather_data_path, picked_country)
        directory_pathc = os.path.join(weather_data_pathc, picked_country)
        
        
        # This will update data for areas if need be replacing with the new file
        if os.path.exists(directory_path):
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                os.remove(file_path)
            for file_name in os.listdir(directory_pathc):
                file_path = os.path.join(directory_pathc, file_name)
                os.remove(file_path)
            os.rmdir(directory_pathc)
            os.rmdir(directory_path)
            print('removed previous contents and will update with the next values')
            
            
        ''' 
            This loop will go through all coordinates of the world to
            select for the the country that has been chosen and collect
            coordinate weather data on that country the data will be 
            saved in a the allocated directories
        '''
        
        # will step through coordinates based on the resolution 
        # larger resolution values indicate faster but less data
        # and vice versa
        for lon in range(-1795, 1800, int(float(lon_res) * 10)):
            for lat in range(-900, 900, int(float(lat_res) * 10)):
                lat1 = float(lat / 10)
                lon1 = float(lon / 10)
                
                
                # check if the coordinate is physical land
                if globe.is_land(lat1, lon1):    
                    
                    coordinates = [(lat1, lon1)]
                    
                    # This will find the location address of the coordinate
                    # we then select just the country value
                    data = reverse_geocode.search(coordinates)
                    country = data[0]['country']
                    
                    # Output to console that we will be aquiring data for this country
                    if (picked_country == country):
                        print('Acquiring coordinate data ' + str(lat1) + "," + str(lon1) + " " + country)
                        output.append('Acquiring coordinate data ' + str(lat1) + "," + str(lon1) + " " + country)
                        
                        
                        # will create a new directory for folders/countries that we have never aquired data from
                        if not os.path.exists(directory_path):
                            
                            # Create the directory
                            os.makedirs(directory_path)
                            print(f"New directory '{directory_path}' has been added.")
                                 
                            
                        # File path to save
                        file_name = f"{lat1}_{lon1}.csv"
                        file_path = os.path.join(directory_path, file_name)
                        print(file_path)
                        # Check if file already exists if need be
                        if os.path.exists(file_path):
                            print(f"File {file_name} already exists. Skipping download.")
                            continue
                        
                        # Will now actively try to download the data from visual crossing
                        # Errors will be caught in case of failures
                        try:
                            '''    
                                IMPORTANT if you are going to run this program please sign up for an account at visual crossing  and paste the key into the key variable
                                 To have a practical usage it will cost money however the program may work for smaller queries for territories instead of countries
                                 e.g. Aruba ,St pierre and Miquelon etc.
                            '''
                            key = 'this key has been omitted replace with own key if supscription has been made'
                            ResultBytes = urllib.request.urlopen("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
                                                                + str(lat1) + "%2C%20" + str(lon1) + "/"+str(fromdate)+"/"+str(todate)+"?unitGroup=metric&include=days&key="+key+"&contentType=csv")
                            
                            CSVText = csv.reader(codecs.iterdecode(ResultBytes, 'utf-8'))
                            
                            # Downloading the file
                            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                for row in CSVText:
                                    csv_writer.writerow(row)
                            
                            print(f"File {file_name} downloaded successfully.")
                        
                        except urllib.error.HTTPError as e:
                            ErrorInfo = e.read().decode()
                            print('Error code: ', e.code, ErrorInfo)
                        
                        except urllib.error.URLError as e:
                            ErrorInfo = e.read().decode()
                            print('Error code: ', e.code, ErrorInfo)
        output.append('Download Complete!!')
        return output
            
    
        print('Download Complete!!')
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []