import pandas as pd
import numpy as np
import os

# Chemins
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'Loan_Data.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'data_clean_features.csv')


def load_data(path):
    df = pd.read_csv(path)
    print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def check_quality(df):
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"Valeurs manquantes : {missing}")
    print(f"Doublons : {duplicates}")
    return df


def clean_features(df):
    # Suppression de l'identifiant client, inutile pour le ML
    df = df.drop(columns=['customer_id'])
    print(f"Features retenues : {df.columns.tolist()}")
    return df


def export_data(df, path):
    df.to_csv(path, index=False)
    print(f"Fichier exporté : {path}")
    print(f"Shape finale : {df.shape}")


if __name__ == '__main__':
    print("=== Chargement ===")
    df = load_data(INPUT_FILE)

    print("\n=== Vérification qualité ===")
    df = check_quality(df)

    print("\n=== Nettoyage et sélection des features ===")
    df = clean_features(df)

    print("\n=== Export ===")
    export_data(df, OUTPUT_FILE)

    print("\nPipeline terminé avec succès.")
