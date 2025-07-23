import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Charger les données
@st.cache_data
def load_data():
    # df = pd.read_csv("C:/datamining/cleaned_ingredients.csv")
    # df = pd.read_csv("D:/fullstack/Ai/aifood/cleaned_ingredients.csv")
    df = pd.read_csv("cleaned_ingredients.csv")  # Relative path from app.py

    cols_to_convert = [
        'Energy_kcal', 'Protein_g', 'Saturated_fats_g', 'Fat_g', 
        'Sugar_g', 'VitA_mcg', 'VitD2_mcg', 'VitE_mg'
    ]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)
    return df

df = load_data()

# Fonction de classification
def classify_product(row):
    try:
        energy = float(row['Energy_kcal'])
        protein = float(row['Protein_g'])
        saturated_fats = float(row['Saturated_fats_g'])
        fat = float(row['Fat_g'])
        sugar = float(row['Sugar_g'])
        vit_a = float(row['VitA_mcg'])
        vit_d = float(row['VitD2_mcg'])
        vit_e = float(row['VitE_mg'])
        
        criteria = [
            energy < 500,
            fat < 25,
            sugar < 18,
            protein > 10,
            saturated_fats < 10,
            vit_a >= 900,
            vit_d >= 150,
            vit_e >= 150
        ]
        
        score = sum(criteria)
        confidence = (score / 8) * 100
        
        return pd.Series([
            'Good' if score >= 5 else 'Bad', 
            confidence
        ])
    
    except Exception as e:
        return pd.Series(['Error', 0])

# Appliquer la classification
df[['Target', 'Confidence']] = df.apply(classify_product, axis=1)
df = df[df['Target'] != 'Error']

# Préparation des données
features = ['Energy_kcal', 'Protein_g', 'Saturated_fats_g', 
            'Fat_g', 'Sugar_g', 'VitA_mcg', 'VitD2_mcg', 'VitE_mg']
X = df[features]
y = df['Target']

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraîner un modèle (Random Forest par exemple)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Interface Streamlit
st.title("Classification Nutritionnelle et Suggestions d'Amélioration")

# Saisie des valeurs nutritionnelles
st.header("Entrez les valeurs nutritionnelles")
energy = st.number_input("Énergie (kcal)", value=200)
protein = st.number_input("Protéines (g)", value=10)
saturated_fats = st.number_input("Graisses saturées (g)", value=5)
fat = st.number_input("Graisses totales (g)", value=15)
sugar = st.number_input("Sucres (g)", value=10)
vit_a = st.number_input("Vitamine A (mcg)", value=900)
vit_d = st.number_input("Vitamine D (mcg)", value=150)
vit_e = st.number_input("Vitamine E (mg)", value=150)

# Bouton pour faire la prédiction
if st.button("Prédire"):
    input_data = np.array([[energy, protein, saturated_fats, fat, sugar, vit_a, vit_d, vit_e]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    
    st.write(f"Résultat : {prediction}")
    st.write(f"Probabilités: [Bad: {probabilities[0]:.2%}, Good: {probabilities[1]:.2%}]")
    
    if prediction == 'Bad':
        st.write("Colonnes à améliorer pour passer à 'Good':")
        weak_points = []
        if energy >= 500:
            weak_points.append("Énergie (kcal) : Réduire en dessous de 500 kcal")
        if fat >= 25:
            weak_points.append("Graisses totales (g) : Réduire en dessous de 25 g")
        if sugar >= 18:
            weak_points.append("Sucres (g) : Réduire en dessous de 18 g")
        if protein <= 10:
            weak_points.append("Protéines (g) : Augmenter au-dessus de 10 g")
        if saturated_fats >= 10:
            weak_points.append("Graisses saturées (g) : Réduire en dessous de 10 g")
        if vit_a < 900:
            weak_points.append("Vitamine A (mcg) : Augmenter au-dessus de 900 mcg")
        if vit_d < 150:
            weak_points.append("Vitamine D (mcg) : Augmenter au-dessus de 150 mcg")
        if vit_e < 150:
            weak_points.append("Vitamine E (mg) : Augmenter au-dessus de 150 mg")
        
        if weak_points:
            for point in weak_points:
                st.write(f"- {point}")
        else:
            st.write("Aucune amélioration nécessaire (erreur de prédiction).")