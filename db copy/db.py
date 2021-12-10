from flask import Flask
from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///datatrain.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    place = db.Column(db.String(120), nullable=False)
    history = db.Column(db.String(360), nullable=False)
    images = db.Column(db.Boolean, nullable=False)
    video = db.Column(db.Boolean, nullable=False)
    model_3d = db.Column(db.Boolean, nullable=False)
    render_path = db.Column(db.Boolean, nullable=False)

    model_id = db.Column(db.Integer, db.ForeignKey('model.id'),
        nullable=False)
    model = db.relationship('Model',
        backref=db.backref('profiles', lazy=True))
   
    def __repr__(self):
        return '<Profile %r>' % self.rg_model


class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(80), nullable=False)
    description = db.Column(db.String(80), nullable=False)
    bucket = db.Column(db.String(80), nullable=False)
    type = db.Column(db.String(80), nullable=False)
    status = db.Column(db.String(80), nullable=False)
    process = db.Column(db.String(80), nullable=False)
    checkpoint = db.Column(db.String(120), nullable=False)
    last_test= db.Column(db.String(120), nullable=False)
    last_step= db.Column(db.String(120), nullable=False)
    max_step= db.Column(db.String(120), nullable=False)
    path_render = db.Column(db.String(120), nullable=False)
    time_train = db.Column(db.String(120), nullable=False)
    time_render = db.Column(db.String(120), nullable=False)
    config = db.Column(db.String(120), nullable=False)

    
    
    def __repr__(self):
        return '<Model %r>' % self.rg_model

class Train(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    last_step= db.Column(db.String(120), nullable=False)
    i_loss= db.Column(db.String(120), nullable=False)
    avg_loss= db.Column(db.String(120), nullable=False)
    weight_l2= db.Column(db.String(120), nullable=False)
    lr= db.Column(db.String(120), nullable=False)
    rays_per_sec= db.Column(db.String(120), nullable=False)

    model_id = db.Column(db.Integer, db.ForeignKey('model.id'),
        nullable=False)
    model = db.relationship('Model',
        backref=db.backref('trains', lazy=True))

    
    def __repr__(self):
        return '<Train %r>' % self.rg_model

class Eval(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    psnr= db.Column(db.String(120), nullable=False)
    ssim= db.Column(db.String(120), nullable=False)
    cpu_percent= db.Column(db.String(120), nullable=False)
    mem_percent= db.Column(db.String(120), nullable=False)
    eval= db.Column(db.String(120), nullable=False)
    eval_path = db.Column(db.String(120), nullable=False)

    model_id = db.Column(db.Integer, db.ForeignKey('model.id'),
        nullable=False)
    model = db.relationship('Model',
        backref=db.backref('evals', lazy=True))

    
    def __repr__(self):
        return '<Eval %r>' % self.rg_model

class Render(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type= db.Column(db.String(120), nullable=False)
    n_images= db.Column(db.String(120), nullable=False)
    cpu_percent= db.Column(db.String(120), nullable=False)
    mem_percent= db.Column(db.String(120), nullable=False)
    render_path= db.Column(db.String(120), nullable=False)

    model_id = db.Column(db.Integer, db.ForeignKey('model.id'),
        nullable=False)
    model = db.relationship('Model',
        backref=db.backref('renders', lazy=True))

    
    def __repr__(self):
        return '<Render %r>' % self.rg_model