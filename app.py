from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from flask import Flask, render_template, request, redirect, url_for, flash, session
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from flask import send_file

from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired


app = Flask(__name__)
# Secret Key for Security
app.secret_key = "8666f8dfcf62158faf17f6e8ecb79830"
app.config['WTF_CSRF_ENABLED'] = False

# Ensure Upload Folder Exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

PLOT_FOLDER = "static/plots"
os.makedirs(PLOT_FOLDER, exist_ok= True)
app.config["PLOT_FOLDER"] = PLOT_FOLDER

# Store DataFrame in Session
data_storage = {}



# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


# Route for Load Data Page
@app.route('/load_data', methods=['GET', 'POST'])
def load_data():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file selected!", "danger")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No file selected!", "warning")
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['file_path'] = filepath  # Save filepath in session

            # Load Data into Pandas DataFrame
            df = pd.read_csv(filepath, encoding="utf-8")
            data_storage['df'] = df  # Store DataFrame in memory

            flash("Dataset loaded successfully!", "success")

    return render_template('load_data.html')


# Route for Previewing Sample Data
@app.route('/preview_data')
def preview_data():
    df = data_storage.get('df', None)
    if df is None:
        return "No data loaded yet!", 400

    sample_data = df.head(5).to_html(classes="table table-striped table-bordered")
    return sample_data  # Returning HTML table


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    df = data_storage.get('df', None)

    if df is None:
        flash("No dataset loaded. Please load a dataset first!", "warning")
        return redirect(url_for('load_data'))

    if request.method == 'POST':
        # Drop duplicate rows if any
        df = df.drop_duplicates()

        # Drop columns with excessive missing values (more than 50%)
        df = df.dropna(thresh=len(df) * 0.5, axis=1)

        # Fill missing values with column mean (for numeric columns)
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col].fillna(df[col].mean(), inplace=True)

        # Store preprocessed DataFrame
        data_storage['df'] = df

        flash("Preprocessing completed successfully!", "success")

    return render_template('preprocess.html')


@app.route('/preview_preprocessed')
def preview_preprocessed():
    df = data_storage.get('df', None)
    if df is None:
        return "No preprocessed data available!", 400

    sample_data = df.head(10).to_html(classes="table table-striped table-bordered")
    return sample_data  # Returning HTML table


@app.route('/visualize')
def visualize():
    df = data_storage.get('df', None)

    if df is None:
        flash("No dataset loaded. Please load a dataset first!", "warning")
        return redirect(url_for('load_data'))

    return render_template('visualize.html')


@app.route('/generate_plot/<plot_type>')
def generate_plot(plot_type):
    df = data_storage.get('df', None)

    if df is None:
        return "No dataset loaded!", 400

    # Sample the data to reduce size
    df_sample = df.sample(n=min(500, len(df)))  # Take max 500 rows

    plt.figure(figsize=(8, 5))

    if plot_type == "departure_delay":
        sns.histplot(df_sample["departure_delay"].dropna(), bins=20, kde=True, color='blue')
        plt.title("Distribution of Departure Delays")
        plt.xlabel("Minutes Delayed")
        plt.ylabel("Frequency")

    elif plot_type == "arrival_delay":
        sns.histplot(df_sample["arrival_delay"].dropna(), bins=20, kde=True, color='red')
        plt.title("Distribution of Arrival Delays")
        plt.xlabel("Minutes Delayed")
        plt.ylabel("Frequency")

    elif plot_type == "weather_vs_delay":
        sns.scatterplot(x=df_sample["HourlyPrecipitation_x"].fillna(0),
                        y=df_sample["departure_delay"].fillna(0),
                        alpha=0.5)
        plt.title("Weather Precipitation vs Departure Delay")
        plt.xlabel("Precipitation (mm)")
        plt.ylabel("Departure Delay (min)")

    elif plot_type == "airport_vs_delay":
        top_airports = df_sample.groupby("origin_airport")["departure_delay"].mean().nlargest(10)
        top_airports.plot(kind="bar", color="green")
        plt.title("Top 10 Airports with Highest Average Departure Delays")
        plt.xlabel("Airport")
        plt.ylabel("Avg Delay (min)")
        plt.xticks(rotation=45)

    else:
        return "Invalid plot type!", 400

    plot_path = os.path.join("static", f"{plot_type}.png")
    plt.savefig(plot_path)
    plt.close()

    return send_file(plot_path, mimetype='image/png')


@app.route('/train_model', methods=['POST'])
def train_model():
    df = data_storage.get('df', None)

    if df is None:
        flash("No dataset loaded. Please load a dataset first!", "warning")
        return redirect(url_for('load_data'))

    # Convert delay into categorical labels
    df['delay_status'] = df['arrival_delay'].apply(lambda x: 1 if x > 15 else 0)  # 1 = Delayed, 0 = On Time

    # Select Features (Only relevant numerical columns)
    features = ['scheduled_elapsed_time', 'HourlyDryBulbTemperature_x', 'HourlyPrecipitation_x',
                'HourlyVisibility_x', 'HourlyWindSpeed_x']
    X = df[features].fillna(0)  # Fill NaN values
    y = df['delay_status']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = {}
    classification_reports = {}

    best_model_name = None
    best_accuracy = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)  # Store report as dict

        results[name] = accuracy
        classification_reports[name] = report

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            joblib.dump(model, "models/best_model.pkl")

    # Store results in session
    session['model_results'] = results
    session['classification_reports'] = classification_reports
    session['best_model'] = best_model_name

    flash(f"Model training completed! 🎉 Best model: {best_model_name} with {best_accuracy:.4f} accuracy.", "success")

    return redirect(url_for('model'))


@app.route('/model')
def model():
    results = session.get('model_results', None)
    return render_template('model.html', results=results)


class PredictionForm(FlaskForm):
    scheduled_elapsed_time = FloatField("Scheduled Elapsed Time (min)", validators=[DataRequired()])
    temperature = FloatField("Temperature (°C)", validators=[DataRequired()])
    precipitation = FloatField("Precipitation (mm)", validators=[DataRequired()])
    visibility = FloatField("Visibility (km)", validators=[DataRequired()])
    wind_speed = FloatField("Wind Speed (km/h)", validators=[DataRequired()])
    submit = SubmitField("Predict")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictionForm()
    model_path = "models/best_model.pkl"
    prediction = None  # Default value

    if request.method == 'POST':  # Ensure form submission is detected
        print("✅ Form submitted!")  # Debug Check 1

        if form.validate():  # Validate form manually
            print("✅ Form validation successful!")  # Debug Check 2

            # Load trained model
            try:
                model = joblib.load(model_path)
                print("✅ Model loaded successfully!")  # Debug Check 3
            except FileNotFoundError:
                flash("⚠ Model not found! Please train the model first.", "danger")
                print("❌ Model file is missing!")  # Debug Check 4
                return redirect(url_for('model'))

            # Collect user input as a list
            input_data = [[
                float(form.scheduled_elapsed_time.data),
                float(form.temperature.data),
                float(form.precipitation.data),
                float(form.visibility.data),
                float(form.wind_speed.data)
            ]]
            print(f"✅ Input Data: {input_data}")  # Debug Check 5

            # Make prediction
            pred = model.predict(input_data)
            print(f"✅ Model Prediction: {pred}")  # Debug Check 6

            prediction = "Delayed" if pred[0] == 1 else "On Time"
            print(f"✅ Final Prediction: {prediction}")  # Debug Check 7

            flash(f"Prediction: {prediction}", "success")

    return render_template('predict.html', form=form, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)