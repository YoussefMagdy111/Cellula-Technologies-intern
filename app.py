from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --- Custom transformer placeholder for pipeline compatibility ---
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method="zscore", threshold=3):
        self.method = method
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# --- Load trained pipeline ---
model = joblib.load("model.pkl")

# --- Flask app ---
app = Flask(__name__)

# --- HTML Template ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hotel Booking Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1d976c, #93f9b9);
            margin: 0; padding: 0;
        }
        .container {
            width: 600px;
            margin: 60px auto;
            background: #fff;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.25);
        }
        h2 {
            text-align: center;
            color: #1d976c;
            margin-bottom: 20px;
        }
        form label {
            display: block;
            margin-top: 12px;
            font-weight: bold;
            color: #333;
        }
        form input, form select {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 14px;
        }
        button {
            width: 100%;
            padding: 12px;
            margin-top: 20px;
            background: #1d976c;
            border: none;
            border-radius: 6px;
            color: #fff;
            font-size: 16px;
            cursor: pointer;
            transition: 0.3s;
        }
        button:hover {
            background: #158a5b;
        }
        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 8px;
            font-size: 20px;
            text-align: center;
            font-weight: bold;
        }
        .confirmed {
            background: #e6ffed;
            border-left: 8px solid #28a745;
            color: #155724;
        }
        .canceled {
            background: #ffe6e6;
            border-left: 8px solid #dc3545;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Hotel Booking Prediction</h2>
        <form method="POST">
            <label>Number of Adults:</label>
            <input type="number" name="number_of_adults" required>

            <label>Number of Children:</label>
            <input type="number" name="number_of_children" required>

            <label>Weekend Nights:</label>
            <input type="number" name="number_of_weekend_nights" required>

            <label>Week Nights:</label>
            <input type="number" name="number_of_week_nights" required>

            <label>Type of Meal:</label>
            <select name="type_of_meal">
                <option value="Meal Plan 1">Meal Plan 1</option>
                <option value="Meal Plan 2">Meal Plan 2</option>
                <option value="Meal Plan 3">Meal Plan 3</option>
                <option value="Not Selected">Not Selected</option>
            </select>

            <label>Car Parking Space:</label>
            <input type="number" name="car_parking_space" required>

            <label>Room Type:</label>
            <select name="room_type">
                <option value="Room_Type 1">Room_Type 1</option>
                <option value="Room_Type 2">Room_Type 2</option>
                <option value="Room_Type 3">Room_Type 3</option>
                <option value="Room_Type 4">Room_Type 4</option>
            </select>

            <label>Lead Time:</label>
            <input type="number" name="lead_time" required>

            <label>Market Segment:</label>
            <select name="market_segment_type">
                <option value="Offline">Offline</option>
                <option value="Online">Online</option>
                <option value="Corporate">Corporate</option>
            </select>

            <label>Repeated Guest:</label>
            <select name="repeated">
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>

            <label>P-C:</label>
            <input type="number" name="P_C" required>

            <label>P-not-C:</label>
            <input type="number" name="P_not_C" required>

            <label>Average Price:</label>
            <input type="number" step="0.01" name="average_price" required>

            <label>Special Requests:</label>
            <input type="number" name="special_requests" required>

            <label>Date of Reservation:</label>
            <input type="date" name="date_of_reservation" required>

            <button type="submit">Predict</button>
        </form>

        {% if prediction is not none %}
            <div class="result {% if prediction == 'Not_Canceled' %}confirmed{% else %}canceled{% endif %}">
                {% if prediction == 'Not_Canceled' %}
                    Great! Your booking looks confirmed.
                {% elif prediction == 'Canceled' %}
                    Sorry, this booking looks like it might be canceled.
                {% else %}
                    {{ prediction }}
                {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # --- Match training column names exactly ---
            features = {
                "number of adults": [int(request.form["number_of_adults"])],
                "number of children": [int(request.form["number_of_children"])],
                "number of weekend nights": [int(request.form["number_of_weekend_nights"])],
                "number of week nights": [int(request.form["number_of_week_nights"])],
                "type of meal": [request.form["type_of_meal"]],
                "car parking space": [int(request.form["car_parking_space"])],
                "room type": [request.form["room_type"]],
                "lead time": [int(request.form["lead_time"])],
                "market segment type": [request.form["market_segment_type"]],
                "repeated": [int(request.form["repeated"])],
                "P-C": [int(request.form["P_C"])],
                "P-not-C": [int(request.form["P_not_C"])],
                "average price ": [float(request.form["average_price"])],  # keep space if in training
                "special requests": [int(request.form["special_requests"])],
                "date of reservation": [request.form["date_of_reservation"]]
            }

            X = pd.DataFrame(features)

            # --- Predict ---
            pred = model.predict(X)[0]
            print("Raw model prediction:", pred)  # Debugging only

            # --- Handle numeric or string outputs ---
            if str(pred).lower() in ["0", "canceled", "cancelled"]:
                prediction = "Canceled"
            elif str(pred).lower() in ["1", "not_canceled", "not cancelled"]:
                prediction = "Not_Canceled"
            else:
                prediction = str(pred)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(HTML_PAGE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
