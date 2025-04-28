from Weather_app import db,login_manager
from Weather_app import bcrypt
from flask_login import UserMixin
import json

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model,UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    password_hash = db.Column(db.String(length=60), nullable=False)
    username = db.Column(db.String(length=30), nullable=False, unique=True)
    email = db.Column(db.String(length=50), nullable=False)
    queries = db.relationship('Query', backref='user', lazy=True)
    
    def __repr__(self):
        return self.username

    @property
    def password(self):
        return self.password
    
    @password.setter
    def password(self,plain_text_password):
        self.password_hash = bcrypt.generate_password_hash(plain_text_password).decode('utf-8')    

    def check_password_correction(self,attempted_password):
        return bcrypt.check_password_hash(self.password_hash,attempted_password)
            
class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    infotype = db.Column(db.String(60), nullable=False)
    content = db.Column(db.Text, nullable=True,unique=True)
    
    def set_content(self, data):
        self.content = json.dumps(data)

    def get_content(self):
        return json.loads(self.content) if self.content else None
    
    image = db.Column(db.String(255), nullable=True)