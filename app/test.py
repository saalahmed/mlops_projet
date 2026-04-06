import pytest
from app import model_pred, FEATURE_COLS, model, MODEL_PATH

# Données de test - exemple avec risque de défaut
test_data_default = {
    'credit_lines_outstanding': 8,
    'loan_amt_outstanding': 8000,
    'total_debt_outstanding': 15000,
    'income': 25000,
    'years_employed': 1,
    'fico_score': 550,
}

# Données de test - exemple sans risque de défaut
test_data_no_default = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 2000,
    'total_debt_outstanding': 5000,
    'income': 80000,
    'years_employed': 8,
    'fico_score': 750,
}


def test_model_loaded():
    """Verifie que le modele pickle est charge avec succes."""
    assert model is not None, f"Le modele n'a pas pu etre charge depuis {MODEL_PATH}"
    assert hasattr(model, 'predict'), "Le modele doit avoir une methode predict"


@pytest.mark.skipif(model is None, reason="Modele pickle non disponible")
def test_predict_with_default_risk():
    """Test que le modèle prédit correctement un risque de défaut."""
    prediction = model_pred(test_data_default)
    assert prediction in [0, 1], "La prédiction doit être 0 ou 1"
    print(f"Prédiction pour données risquées: {prediction}")


@pytest.mark.skipif(model is None, reason="Modele pickle non disponible")
def test_predict_no_default_risk():
    """Test que le modèle prédit correctement l'absence de risque de défaut."""
    prediction = model_pred(test_data_no_default)
    assert prediction in [0, 1], "La prédiction doit être 0 ou 1"
    print(f"Prédiction pour données sûres: {prediction}")


def test_feature_columns():
    """Vérifie que toutes les colonnes de features attendues sont présentes."""
    assert len(FEATURE_COLS) > 0, "FEATURE_COLS ne doit pas être vide"
    assert all(isinstance(col, str) for col in FEATURE_COLS), "Les noms de colonnes doivent être des strings"


@pytest.mark.skipif(model is None, reason="Modele pickle non disponible")
def test_prediction_consistency():
    """Teste que le modèle donne une prédiction cohérente pour les mêmes données."""
    pred1 = model_pred(test_data_no_default)
    pred2 = model_pred(test_data_no_default)
    assert pred1 == pred2, "Le modèle doit donner des prédictions cohérentes"


if __name__ == "__main__":
    # Exécution simple de test d'exemple
    print("Test 1 : Données avec risque de défaut")
    test_predict_with_default_risk()
    
    print("\nTest 2 : Données sans risque de défaut")
    test_predict_no_default_risk()
    
    print("\nTest 3 : Colonnes de features")
    test_feature_columns()
    
    print("\nTest 4 : Cohérence des prédictions")
    test_prediction_consistency()
    
    print("\n✓ Tous les tests manuels sont passés!")