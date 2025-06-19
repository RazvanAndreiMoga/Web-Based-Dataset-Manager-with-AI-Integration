# app.py
import os
import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)
app.secret_key = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect('/home')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        new_user = User(username=request.form['username'], password=hashed)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(UPLOAD_FOLDER, f"{current_user.username}.csv")
        file.save(filepath)
        session.pop('features', None)
        return redirect('/explore')
    return render_template('upload.html')


def get_df():
    path = os.path.join(UPLOAD_FOLDER, f"{current_user.username}.csv")
    return pd.read_csv(path)

@app.route('/explore')
@login_required
def explore():
    df = get_df()
    return render_template('explore.html', shape=df.shape)

@app.route('/dtypes')
@login_required
def dtypes():
    df = get_df()
    return df.dtypes.to_frame().to_html()

@app.route('/head', methods=['POST'])
@login_required
def head():
    n = int(request.form['n'])
    df = get_df()
    return df.head(n).to_html()

@app.route('/tail', methods=['POST'])
@login_required
def tail():
    n = int(request.form['n'])
    df = get_df()
    return df.tail(n).to_html()

@app.route('/describe')
@login_required
def describe():
    df = get_df()
    return df.describe().to_html()

@app.route('/dropna')
@login_required
def dropna():
    df = get_df().dropna()
    df.to_csv(os.path.join(UPLOAD_FOLDER, f"{current_user.username}.csv"), index=False)
    return redirect('/explore')

@app.route('/select_features', methods=['GET', 'POST'])
@login_required
def select_features():
    df = get_df()
    if request.method == 'POST':
        session['features'] = request.form.getlist('features')
        session['label'] = request.form.get('label')
        return redirect('/transform')
    columns = df.columns.tolist()
    return render_template('select_features.html', columns=columns)

@app.route('/transform', methods=['GET', 'POST'])
@login_required
def transform():
    df = get_df()
    if request.method == 'POST':
        method = request.form['method']
        if method == 'dummies':
            old_features = session['features']
            label = session.get('label')
            df_features = df.drop(columns=[label])
            df_label = df[label]

            df_encoded = pd.get_dummies(df_features)
            df_encoded[label] = df_label  # Add the label back (unmodified)

            # Update features with new encoded columns
            old_features = session['features']
            new_features = []
            for col in old_features:
                if col in df_encoded.columns:
                    new_features.append(col)
                else:
                    new_features.extend([c for c in df_encoded.columns if c.startswith(col + '_')])
            session['features'] = new_features

            df_encoded.to_csv(os.path.join(UPLOAD_FOLDER, f"{current_user.username}.csv"), index=False)
            # Update feature names to match new dummified columns
            new_features = []
            for col in old_features:
                if col in df.columns:
                    new_features.append(col)
                else:
                    # Add all one-hot columns that start with this feature name
                    new_features.extend([c for c in df.columns if c.startswith(col + '_')])
            session['features'] = new_features
        elif method == 'ordinal':
            encoder = OrdinalEncoder()
            df[df.columns] = encoder.fit_transform(df[df.columns])
        df.to_csv(os.path.join(UPLOAD_FOLDER, f"{current_user.username}.csv"), index=False)
        return redirect('/train')
    return render_template('transform.html')

@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    df = get_df()
    features = session.get('features')
    label = session.get('label')
    if not features or not label:
        return "Please select features and a label column first. <a href='/select_features'>Back</a>"

    if request.method == 'POST':
        model_type = request.form['model']
        X = df[features]
        y = df[label]

        if model_type == 'regression':
            model = LinearRegression().fit(X, y)
        elif model_type == 'classification':
            model = RandomForestClassifier().fit(X, y.astype('int'))
        elif model_type == 'clustering':
            model = KMeans(n_clusters=3).fit(X)
            y = model.labels_

        with open(os.path.join(MODEL_FOLDER, f"{current_user.username}.pkl"), 'wb') as f:
            pickle.dump((model, features), f)

        return redirect('/predict')
    return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    model_path = os.path.join(MODEL_FOLDER, f"{current_user.username}.pkl")
    if not os.path.exists(model_path):
        return "Model not found."
    with open(model_path, 'rb') as f:
        model, features = pickle.load(f)
    if request.method == 'POST':
        data = [float(request.form[f]) for f in features]
        pred = model.predict([data])[0]
        return f"<h3>Prediction: {pred}</h3>"
    return render_template('predict.html', features=features)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
