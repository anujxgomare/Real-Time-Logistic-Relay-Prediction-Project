from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load Models
clf_model = joblib.load('model/classifier_model.pkl')
reg_model = joblib.load('model/regressor_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    delay_hours = None
    
    if request.method == 'POST':
        # Extract features from form
        input_features = [
            float(request.form['distance_km']),
            float(request.form['vendor_delay_score']),
            float(request.form['hour_of_day']),
            float(request.form['day_of_week']),
            float(request.form['pickup_delay_minutes']),
            float(request.form['driver_rating']),
            float(request.form['vehicle_age_years']),
            float(request.form['order_weight_kg']),
            int(request.form['num_packages']),
            int(request.form['holiday_flag'])
        ]
        
        input_array = np.array([input_features])

        is_delayed = clf_model.predict(input_array)[0]
        delay_prediction = reg_model.predict(input_array)[0]

        prediction = "YES" if is_delayed == 1 else "NO"
        delay_hours = round(delay_prediction / 60, 2)  # convert minutes to hours

    return render_template('index.html', prediction=prediction, delay_hours=delay_hours)

if __name__ == '__main__':
    app.run(debug=True)
