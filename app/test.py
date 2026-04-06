import unittest
from app import app

class TestCreditSafe(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home(self):
        """Test que la page d'accueil répond bien"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CreditSafe", response.data)

    def test_predict_missing_data(self):
        """Test que l'endpoint predict répond"""
        response = self.client.post("/predict",
            json={
                "credit_lines_outstanding": 3,
                "loan_amt_outstanding": 15000,
                "total_debt_outstanding": 20000,
                "income": 50000,
                "years_employed": 5,
                "fico_score": 700
}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("probabilite_defaut", data)

if __name__ == "__main__":
    unittest.main()