from Weather_app import app,db,data_extractor,data_cleaner,pca_k_means,hashingit,SVM_modeller
from flask import render_template,redirect,url_for,flash,request
from Weather_app.models import User,Query
from Weather_app.forms  import RegisterForm,LoginForm,LocationForm,DataForm,DataExtractionForm
from flask_login import login_user,logout_user,current_user,AnonymousUserMixin
from datetime import datetime
import matplotlib
import time
import os

@app.route("/")
@app.route("/start", methods=['GET', 'POST'])
def start_page():
    print('register mode')
    print(current_user)
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data, email=form.email.data, password=form.password1.data)
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f'account created succesfully you are now logged in as: {user_to_create.username}')            
        return redirect(url_for('unsupervised'))
    
    if form.errors != {}:
        for err_msg in form.errors.values():
            flash(f'there was an error in creating a user: {err_msg}')
    return render_template('start.html', form=form, show_register_form=True, show_login_form=False)

    
@app.route('/login',methods=['GET', 'POST'])
def login():
    print('login mode')
    print(current_user)
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()

        if attempted_user and attempted_user.check_password_correction(
                attempted_password=form.password.data):
            login_user(attempted_user)
            flash(f'Logged in as: {attempted_user.username}',category='success')
            return redirect(url_for('unsupervised'))
        else:
            flash('username and password do not match please try again')
    return render_template('Start.html', show_login_form=True, show_register_form=False, form=form)
        

@app.route('/logout')
def logout_page():
    logout_user()
    flash('you have been logged out',category='info')
    return redirect(url_for('unsupervised'))



@app.route('/unsupervised', methods=['GET', 'POST'])
def unsupervised():
    matplotlib.use('TkAgg')
    print('in unsupervised')
    print(current_user)
    location_form = LocationForm()
    data_form = DataForm()
    image_paths = []
    number = 0  # Initialize the number variable
    # Handle form submission
    
    if data_form.validate_on_submit():
        form_dict = {}
        for field_name, field in data_form._fields.items():
            # Exclude the submit button
            if field_name != 'submit':
                # Use the field_name as the key and the field label as the value
                form_dict[field_name] = field.data

        # Check if the user is logged in
        if not isinstance(current_user, AnonymousUserMixin):
            content = hashingit.hash_it_s(form_dict)
            
            #aquires user object
            user = User.query.filter_by(id=current_user.id).first()
            
            #aquires Query based on if contet matches
            query = Query.query.filter_by(user_id=current_user.id, content=content).first()

            if query:
                image = query.image
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print(image)
                #app.static_folder = query.image
                image_paths = image_array(image)
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-image and app folder  but with user query')
                print(app.static_folder)
                print(image)
                
                mode = 'unsupervised'
                print('we found this query you have done before, loading query up')
                print(image_paths)
                return render_template('process_ui.html', location_form=location_form, data_form=data_form, mode=mode, images=image_paths, form_dict=form_dict,number=number)
            else:
                # Create a new Query object
                start_time = time.time()
                image = pca_k_means.cluster(form_dict)
                end_time = time.time()

                execution_time = end_time - start_time
                print("Execution Time:", execution_time, "seconds")
                #app.static_folder = image
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-image and app folder no user query found')
                print('static_folder')
                print(app.static_folder)
                print('image very important')
                print(image)
                image_paths = image_array(image)
                
                
                
                mode = 'unsupervised'
                new_query = Query(user_id=current_user.id, infotype='unsupervised', content=content,image=image)

                # Add the new Query object to the user's queries
                user.queries.append(new_query)

                # Commit changes to the database
                db.session.add(new_query)
                db.session.commit()

                print(f'No existing query found. Created a new one and saved to the database with directory: {image}')
                print(image_paths)
                return render_template('process_ui.html', location_form=location_form, data_form=data_form, mode=mode, images=image_paths, form_dict=form_dict,number=number)
        else:
            # Cluster the data for anonymous users
            start_time = time.time()
            image = pca_k_means.cluster(form_dict)
            end_time = time.time()
            
            execution_time = end_time - start_time
            print("Execution Time:", execution_time, "seconds")
            #app.static_folder = image
            print(f'returned value from pca_kmeans: {image}')
            print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-image and app folder but with no user ')
            print(app.static_folder)
            print(image)
            
            image_paths = image_array(image)
            mode = 'unsupervised'
            return render_template('process_ui.html', location_form=location_form, data_form=data_form, mode=mode, images=image_paths, form_dict=form_dict,number=number)
    if data_form.errors != {}:
        for err_msg in data_form.errors.values():
            flash(f'there was an error in creating a user: {err_msg}')
    #else:
    #    print('an error has occured')


    # Render the form without pre-filled data for GET requests
    print('good')
    print(image_paths)
    mode = 'unsupervised'
    return render_template('process_ui.html', location_form=location_form, data_form=data_form,images = image_paths,number=number,mode=mode)


def image_array(directory):
    static_folder = 'static'  # Assuming 'static' is the static folder name

    # Remove the absolute path up to the 'static' folder
    relative_path = directory.split(static_folder)[1].strip("\\").strip("/")
    
    array = []
    
    for image in os.listdir(directory):
        image_path = os.path.join(relative_path, image).replace("\\", "/")
        array.append(image_path)

    return array














@app.route('/supervised', methods=['GET','POST'])
def supervised():
    matplotlib.use('Agg')
    print('in supervised')
    print(current_user)
    location_form = LocationForm()
    data_form = DataForm()
    image_paths = []
    number = 0  # Initialize the number variable
    # Handle form submission
    if data_form.validate_on_submit():
        print('unsupervised activated')
        form_dict = {}
        for field_name, field in data_form._fields.items():
            # Exclude the submit button
            if field_name != 'submit':
                # Use the field_name as the key and the field label as the value
                form_dict[field_name] = field.data
        # Check if the user is logged in

        if not isinstance(current_user, AnonymousUserMixin):
            content = hashingit.hash_it_s(form_dict)
            #aquires user object
            user = User.query.filter_by(id=current_user.id).first()
            #aquires Query based on if contet matches
            query = Query.query.filter_by(user_id=current_user.id, content=content).first()
            
            if query:
                image = query.image
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print(image)

                image_paths = image_array(image)

                
                mode = 'supervised'
                print('we found this query you have done before, loading query up')
                print(image_paths)
                return render_template('process_ui.html', location_form=location_form, data_form=data_form, mode=mode, images=image_paths, form_dict=form_dict,number=number)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                # Create a new Query object
                start_time = time.time()

                image = SVM_modeller.predict(form_dict)
                end_time = time.time()
                
                execution_time = end_time - start_time
                print("Execution Time:", execution_time, "seconds")
                #app.static_folder = image
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-image and app folder no user query found')

                print('image very important')
                print(image)
                image_paths = image_array(image)
                
                
                
                mode = 'supervised'
                new_query = Query(user_id=current_user.id, infotype='unsupervised', content=content,image=image)

                # Add the new Query object to the user's queries
                user.queries.append(new_query)

                # Commit changes to the database
                db.session.add(new_query)
                db.session.commit()

                print(f'No existing query found. Created a new one and saved to the database with directory: {image}')
                print(image_paths)
                return render_template('process_ui.html', location_form=location_form, data_form=data_form, mode=mode, images=image_paths, form_dict=form_dict,number=number)
        else:
            # Cluster the data for anonymous users
            start_time = time.time()
            image = SVM_modeller.predict(form_dict)
            end_time = time.time()

            execution_time = end_time - start_time
            print("Execution Time:", execution_time, "seconds")
            #app.static_folder = image
            print(f'returned value from pca_kmeans: {image}')
            print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-image and app folder but with no user ')
            
            print(image)
            
            image_paths = image_array(image)
            mode = 'supervised'
            return render_template('process_ui.html', location_form=location_form, data_form=data_form, mode=mode, images=image_paths, form_dict=form_dict,number=number)

    # Render the form without pre-filled data for GET requests
    print('good')
    mode= 'supervised'
    print('hello anyone home?')
    print(mode)
    print('hello anyone home?')
    print(image_paths)
    return render_template('process_ui.html', location_form=location_form, data_form=data_form,images = image_paths,number=number,mode=mode)







@app.route('/adddata', methods=['GET', 'POST'])
def add_data():
    print('add data')
    print(current_user)
    
    # Forms used in this route
    form = DataExtractionForm()
    output = []
    
    # if the submitted form has had all its inputs succesfully validated woth not constraints bring broken
    if form.validate_on_submit():
        print('activate')

        # Redirect the standard output to capture the print statements
        output = data_extractor.Extract(form.lat_resolution.data,
                                        form.lon_resolution.data,
                                        form.country.data,
                                        form.fromdate.data,
                                        form.todate.data)
        print('Starting Data Cleaning')
        output.append('Starting Data Cleaning')
        final_output = data_cleaner.process_country_folder(form.country.data,output)
    
    
            
        return render_template('add_data.html', form=form, output=final_output)

    # If form validation fails, flash a message
    if form.errors != {}:
        for err_msg in form.errors.values():
            flash(f'Form validation failed. Please check the form inputs: {err_msg}')

    # Ensure that the output variable is defined even if form validation fails
    output = []

    return render_template('add_data.html', form=form, output=output)
   
    
