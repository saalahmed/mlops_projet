from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd

app = Flask("CreditSafe")

with open("app/model.pkl", "rb") as f:
    pipeline = pickle.load(f)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>CreditSafe</title>
    <style>
        body { font-family: Arial; max-width: 500px; margin: 50px auto; padding: 20px; }
        h1 { color: #2c3e50; text-align: center; }
        input { width: 100%; padding: 8px; margin: 5px 0 15px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { width: 100%; padding: 12px; background: #2c3e50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #34495e; }
        #result { margin-top: 20px; padding: 15px; border-radius: 4px; text-align: center; font-size: 18px; }
        .danger { background: #ffe0e0; color: #c0392b; }
        .safe { background: #e0ffe0; color: #27ae60; }
        label { font-weight: bold; color: #555; }
    </style>
</head>
<body>
    <h1>🏦 CreditSafe</h1>
    <p style="text-align:center; color:#888;">Prédiction de défaut de prêt</p>
    <label>Lignes de crédit en cours</label>
    <input type="number" id="credit_lines_outstanding" placeholder="ex: 3">
    <label>Montant du prêt en cours</label>
    <input type="number" id="loan_amt_outstanding" placeholder="ex: 15000">
    <label>Dette totale en cours</label>
    <input type="number" id="total_debt_outstanding" placeholder="ex: 20000">
    <label>Revenu annuel</label>
    <input type="number" id="income" placeholder="ex: 50000">
    <label>Années d'emploi</label>
    <input type="number" id="years_employed" placeholder="ex: 5">
    <label>Score FICO</label>
    <input type="number" id="fico_score" placeholder="ex: 700">
    <button onclick="predict()">Analyser le risque</button>
    <div id="result"></div>
    <script>
        async function predict() {
            const data = {
                credit_lines_outstanding: parseFloat(document.getElementById('credit_lines_outstanding').value),
                loan_amt_outstanding: parseFloat(document.getElementById('loan_amt_outstanding').value),
                total_debt_outstanding: parseFloat(document.getElementById('total_debt_outstanding').value),
                income: parseFloat(document.getElementById('income').value),
                years_employed: parseFloat(document.getElementById('years_employed').value),
                fico_score: parseFloat(document.getElementById('fico_score').value)
            };
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const result = await response.json();
            const div = document.getElementById('result');
            if (result.prediction === 1) {
                div.className = 'danger';
                div.innerHTML = 'Risque de défaut détecté<br><small>Probabilité: ' + (result.probabilite_defaut * 100).toFixed(1) + '%</small>';
            } else {
                div.className = 'safe';
                div.innerHTML = 'Pas de risque détecté<br><small>Probabilité: ' + (result.probabilite_defaut * 100).toFixed(1) + '%</small>';
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    FEATURES = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    df = pd.DataFrame([data])[FEATURES]
    prediction = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0][1]
    return jsonify({
        "prediction": int(prediction),
        "probabilite_defaut": round(float(proba), 4)
    })

if __name__ == "__main__":
    app.run(debug=True)