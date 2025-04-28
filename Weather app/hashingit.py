import hashlib

'''
    each function has the same purpose of creating a unique string based on the 
    form data, this will allow for users to be able to recollect query previously
    made to remove wasteful proccesing

'''

def hash_it_s(form):
    # Extract and alphabetize location information
    variables_string = form['location']
    variables_list = variables_string.split(',')
    
    # Remove the last empty string (resulting from the trailing comma)
    if variables_list[-1] == '':
        variables_list.pop()
    
    # Alphabetize the list
    variables_list = sorted(variables_list)
    variables_string = ','.join(variables_list)
    
    # Convert date values to strings
    todate = str(form['todate'])
    fromdate = str(form['fromdate'])
    
    # Variable names
    variable_names = ['temp', 'tempmin', 'tempmax', 'humidity', 'precip', 'precipprob', 'precipcover', 'snowdepth', 'windspeed', 'cloudcover', 'solarenergy', 'uvindex']
    
    # Remove values in the array where the corresponding value in the form is None
    filtered_variable_names = [attribute for attribute in variable_names if form.get(attribute) is not None]
    
    # Get key-value pairs for each attribute
    attribute_key_values = [f"{attribute}:{form[attribute]}" for attribute in filtered_variable_names]
    
    # Combine all information into a string
    combined_info = f"{variables_string}-{todate}-{fromdate}-{','.join(attribute_key_values)}"
    
    # Hash the combined information
    hash_value = hashlib.md5(combined_info.encode()).hexdigest()
    
    return hash_value


def hash_it_u(form):
    # Extract and alphabetise location information
    variables_string = form['location']
    variables_list = variables_string.split(',')
    
    # Remove the last empty string (resulting from the trailing comma)
    if variables_list[-1] == '':
        variables_list.pop()
    
    # Alphabetize the list
    variables_list = sorted(variables_list)
    variables_string = ','.join(variables_list)
    # Convert date values to strings
    todate = str(form['todate'])
    fromdate = str(form['fromdate'])
    
    # Define variable names
    variable_names = ['temp', 'tempmin', 'tempmax', 'humidity', 'precip', 'precipprob', 'precipcover', 'snowdepth', 'windspeed', 'cloudcover', 'solarenergy', 'uvindex']
    
    # Remove values in the array where the corresponding value in the form is None
    filtered_variable_names = [attribute for attribute in variable_names if form.get(attribute) is not None]
    combined_info = f"{variables_string}-{todate}-{fromdate}-{','.join(filtered_variable_names)}"
    
    # Hash the combined information
    hash_value = hashlib.md5(combined_info.encode()).hexdigest()
    return hash_value
