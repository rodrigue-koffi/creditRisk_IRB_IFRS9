import pandas as pd
import sys
sys.path.append('.')

# Test 1: Charger depuis raw
file_path = 'data/raw/german_credit_data.xlsx'
print(f"Tentative de chargement: {file_path}")

try:
    df = pd.read_excel(file_path, sheet_name='german_credit_data(1)')
    print(f"✅ Succès! {len(df)} lignes chargées")
    print(f"Colonnes: {df.columns.tolist()[:5]}...")
except Exception as e:
    print(f"❌ Erreur: {e}")
