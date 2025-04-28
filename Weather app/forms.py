from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,DecimalField,IntegerField,SubmitField,SelectField,DateField,BooleanField
from wtforms.validators import Length,EqualTo,Email,DataRequired,ValidationError,Optional,NumberRange
import os
from Weather_app.models import User

class RegisterForm(FlaskForm):
    
    def validate_username(self,username_to_check):
        user = User.query.filter_by(username=username_to_check.data).first()
        if user:
            raise ValidationError('Username already exist try another name')
            
    
    def validate_email(self,email_to_check):
        email = User.query.filter_by(email=email_to_check.data).first()
        if email:
            raise ValidationError('email already exist try another name')        
    
    username=StringField(label='username',validators=[Length(min=2,max=30),DataRequired()])
    email=StringField(label='email',validators=[Email(),DataRequired()])
    password1= PasswordField(label='password',validators=[Length(min=6),DataRequired()])
    password2= PasswordField(label='verify password',validators=[EqualTo('password1'),DataRequired()])
    submit = SubmitField(label='submit')
    
class LoginForm(FlaskForm):
    username=StringField(label='username',validators=[DataRequired()])
    password= PasswordField(label='password',validators=[DataRequired()])
    submit = SubmitField(label='submit')

base_directory = 'Weather_app\cleaned_data'
subdirectories = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]
print(subdirectories)
class LocationForm(FlaskForm):
    location_choice = SelectField(label='Select Directory', choices=subdirectories, default='',coerce=str)  


class DataForm(FlaskForm):
    temp = DecimalField('Temp', validators=[NumberRange(min=-90,max=55),Optional()])
    tempmin = DecimalField('Tempmin', validators=[NumberRange(min=-90,max=55),Optional()])
    tempmax = DecimalField('Tempmax', validators=[NumberRange(min=-90,max=55),Optional()])
    humidity = DecimalField('Humidity', validators=[NumberRange(min=0,max=100),Optional()])
    precip = DecimalField('Precipitation', validators=[NumberRange(min=0, max=12000), Optional() ])
    precipprob = IntegerField('Precipitation Probability', validators=[NumberRange(min=0,max=100),Optional()])
    precipcover = DecimalField('Precipitation Cover', validators=[NumberRange(min=0,max=100),Optional()])
    snowdepth = DecimalField('Snow Depth', validators=[NumberRange(min=-90,max=55),Optional()])
    windspeed = DecimalField('Wind Speed',validators=[NumberRange(min=0,max=410),Optional()])
    cloudcover = DecimalField('Cloud Cover', validators=[NumberRange(min=0,max=100),Optional()])
    solarenergy = DecimalField('Solar Energy', validators=[NumberRange(min=0,max=7800),Optional()])
    uvindex = IntegerField('UV Index', validators=[NumberRange(min=0,max=10),Optional()])
    
    def validate(self, extra_validators=None):
        rv = super().validate(extra_validators=extra_validators)
        if not rv:
            return False
    
        # Check at least one weather attribute filled
        field_names = ['temp', 'tempmin', 'tempmax', 'humidity', 'precip',
                       'precipprob', 'precipcover', 'snowdepth',
                       'windspeed', 'cloudcover', 'solarenergy', 'uvindex']
    
        if not any(getattr(self, field_name).data is not None for field_name in field_names):
            field = getattr(self, 'temp')
            field.errors.append('At least one of the weather attributes must be filled in.')
            return False
    
        return True

    
    
    fromdate = DateField(label='from', validators=[DataRequired()])
    todate = DateField(label='to', validators=[DataRequired()])
    
    def validate_todate(self, field):
        if self.fromdate.data and field.data:
            # Calculate the difference in days between the two dates
            date_difference = (field.data - self.fromdate.data).days

            # Check if the dates are more than or equal to a year apart
            if date_difference >= 365:
                raise ValidationError('The dates should not be more than a year apart.')
       

    def validate_fromdate(self, field):
        if self.todate.data and field.data:
            if field.data >= self.todate.data:
                raise ValidationError('The "from" date must be before the "to" date.')

    
    
    
    location = StringField(label='location')

    def validate_location(self, field):
        if field.data == '':
            raise ValidationError('Select at least 1 country')


    cluster_by_country = BooleanField('Cluster by Country', default=False)
    cluster_by_weather = BooleanField('Cluster by Weather', default=False)    


    def validate_cluster_by_country(self, field):
        if False == field.data and self.cluster_by_weather.data == False:
            print('heres country')
            raise ValidationError('To continue choose between "Cluster by Country" or "Cluster by Weather" ')

    def validate_cluster_by_weather(self, field):
        print('heres weather')
        if True == field.data and self.cluster_by_country.data == True:
            raise ValidationError('Select only one between "Cluster by Country" or "Cluster by Weather" ')



    submit = SubmitField(label='run program')
    
    #111.321
    
    
    
class DataExtractionForm(FlaskForm):
    lat_resolution = SelectField(label='Select Latitude Resolution',
        choices=[
            ('', 'Resolution Value'),  # Add an empty option as the first choice
            ('0.1', '0.1'),('0.2', '0.2'),('0.3', '0.3'),('0.4', '0.4'),
            ('0.5', '0.5'),('0.6', '0.6'),('0.8', '0.8'),('0.9', '0.9'),
            ('1.0', '1.0')
        ], default=None)
    
    lon_resolution = SelectField(label='Select Longitude Resolution',
        choices=[
            ('', 'Resolution Value'),  # Add an empty option as the first choice
            ('0.1', '0.1'),('0.2', '0.2'),('0.3', '0.3'),('0.4', '0.4'),
            ('0.5', '0.5'),('0.6', '0.6'),('0.8', '0.8'),('0.9', '0.9'),
            ('1.0', '1.0')
        ], default=None)

    def validate_lat_resolution(self, field):
        if field.data == '':
            raise ValidationError('you need to add a value to Latitude reolution')
            
    def validate_lon_resolution(self, field):
        if field.data == '':
            raise ValidationError('you need to add a value to Longitude reolution')



    country = SelectField(label='select a country', validators=[DataRequired()],
                          choices=['Afghanistan', 'Aland Islands', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bonaire, Saint Eustatius and Saba', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Christmas Island', 'Cocos (Keeling) Islands', 'Colombia', 'Comoros', 'Congo', 'Congo, The Democratic Republic of the', 'Cook Islands', 'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Falkland Islands (Malvinas)', 'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran, Islamic Republic of', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', "Korea, Democratic People's Republic of", 'Korea, Republic of', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libyan Arab Jamahiriya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico', 'Micronesia, Federated States of', 'Moldova, Republic of', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island', 'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestinian Territory', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Reunion', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Bartelemey', 'Saint Helena', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin', 'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Georgia and the South Sandwich Islands', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Svalbard and Jan Mayen', 'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan', 'Tajikistan', 'Tanzania, United Republic of', 'Thailand', 'Timor-Leste', 'Togo', 'Tokelau', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Virgin Islands, British', 'Virgin Islands, U.S.', 'Wallis and Futuna', 'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe'])
    
  
    fromdate = DateField(label='from', validators=[DataRequired()])
    todate = DateField(label='to', validators=[DataRequired()])
       

    def validate_fromdate(self, field):
        if self.todate.data and field.data:
            if field.data >= self.todate.data:
                raise ValidationError('The "from" date must be before the "to" date.')


    submit = SubmitField(label='run program')