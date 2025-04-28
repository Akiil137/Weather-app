from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

# creating flask app
app = Flask(__name__)

# configuration 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Weather_app.db'  # SQLite database URI
app.config['SECRET_KEY'] = '46c9ad803d608942636b8c73'  # secret key 

# initializing extensions
db = SQLAlchemy(app)  # SQLAlchemy for database ORM
bcrypt = Bcrypt(app)  # bcrypt for password hashing
login_manager = LoginManager(app)  # for user authentication

# Importing routes
from Weather_app import routes  # Importing routes module from Weather_app package
