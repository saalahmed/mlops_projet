from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Chemin du modele sauvegarde
MODEL_PATH = Path(__file__).parent / "model.pkl"

# Features attendées (dans le même ordre que l'entraînement)
# Features attendues (dans le même ordre que l'entraînement)
FEATURE_COLS = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score'
]


def load_model():
    """Charge le modele depuis le fichier pickle."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Modele charge depuis {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"Erreur: Le fichier {MODEL_PATH} n'existe pas")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement du modele: {e}")
        return None


model = load_model()


def model_pred(features):
    """
    Effectue une prédiction basée sur les features fournies.
    
    Args:
        features: dict avec les clés correspondant aux colonnes attendues
        
    Returns:
        int: 0 (pas de défaut) ou 1 (risque de défaut)
    """
    # Créer un DataFrame avec l'ordre correct des colonnes
    test_data = pd.DataFrame([features])[FEATURE_COLS]
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
                        # Recuperation des donnees du formulaire
            features = {
                'credit_lines_outstanding': float(request.form.get('credit_lines_outstanding', 0)),
                'loan_amt_outstanding': float(request.form.get('loan_amt_outstanding', 0)),
                'total_debt_outstanding': float(request.form.get('total_debt_outstanding', 0)),
                'income': float(request.form.get('income', 0)),
                'years_employed': int(request.form.get('years_employed', 0)),
                'fico_score': int(request.form.get('fico_score', 0)),
            }
            
            # Prédiction
            prediction = model_pred(features)
            
            if prediction == 1:
                prediction_text = "⚠️ Risque de défaut détecté. Veuillez revoir les conditions de prêt."
            else:
                prediction_text = "✓ Pas de risque de défaut détecté. Vous pouvez accepter ce prêt."
            
            return render_template("index.html", prediction_text=prediction_text)
        
        except Exception as e:
            return render_template(
                "index.html",
                prediction_text=f"Erreur lors de la prédiction: {str(e)}"
            )
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
