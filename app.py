from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained XGBoost model
model = joblib.load("churn_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None

    if request.method == "POST":

        # -------- Numerical Inputs --------
        age = float(request.form["Age"])
        tenure = float(request.form["Tenure"])
        usage = float(request.form["Usage_Frequency"])
        support = float(request.form["Support_Calls"])
        delay = float(request.form["Payment_Delay"])
        total_spend = float(request.form["Total_Spend"])
        last_interaction = float(request.form["Last_Interaction"])

        # -------- Categorical Inputs --------
        gender = request.form["Gender"]
        subscription = request.form["Subscription_Type"]
        contract = request.form["Contract_Length"]

        # -------- Encoding (MATCH DATASET LOGIC) --------
        # Gender
        gender_enc = 1 if gender == "Male" else 0

        # Subscription Type
        if subscription == "Basic":
            subscription_enc = 0
        elif subscription == "Standard":
            subscription_enc = 1
        else:  # Premium
            subscription_enc = 2

        # Contract Length
        if contract == "Monthly":
            contract_enc = 0
        elif contract == "Quarterly":
            contract_enc = 1
        else:  # Annual
            contract_enc = 2

        # -------- Feature Order MUST MATCH TRAINING --------
        features = np.array([[
            age,
            gender_enc,
            tenure,
            usage,
            support,
            delay,
            subscription_enc,
            contract_enc,
            total_spend,
            last_interaction
        ]])

        # -------- Prediction --------
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        result = "Churn" if pred == 1 else "Not Churn"
        probability = round(prob * 100, 2)

    return render_template("index.html", result=result, probability=probability)


if __name__ == "__main__":
    app.run(debug=True)
