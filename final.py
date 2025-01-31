import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Charger les données
#path salah
# df = pd.read_csv("C:/datamining/cleaned_ingredients.csv")
#path oubey
df = pd.read_csv("C:/Users/j/Desktop/full stack/Ai/Ai-food/cleaned_ingredients.csv")

# Nettoyage des données
cols_to_convert = [
    'Energy_kcal', 'Protein_g', 'Saturated_fats_g', 'Fat_g', 
    'Sugar_g', 'VitA_mcg', 'VitD2_mcg', 'VitE_mg'
]

for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remplacer les valeurs manquantes/invalides par 0
df.fillna(0, inplace=True)

# Fonction de classification avec gestion d'erreur
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
        
        # Critères AJR
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
        print(f"Erreur ligne {row.name}: {str(e)}")
        return pd.Series(['Error', 0])

# Application de la classification
df[['Target', 'Confidence']] = df.apply(classify_product, axis=1)

# Supprimer les erreurs éventuelles
df = df[df['Target'] != 'Error']

# Visualisation de la répartition des classes (Good vs Bad)
plt.figure(figsize=(8, 6))
# sns.countplot(x='Target', data=df, palette='Set2')
sns.countplot(x='Target', data=df, hue='Target', palette='Set2', legend=False)
plt.title("Répartition des classes (Good vs Bad)")
plt.xlabel("Classe")
plt.ylabel("Nombre d'aliments")
plt.show()

# Afficher le nombre d'aliments par classe
print("\nRépartition des classes :")
print(df['Target'].value_counts())

# Préparation des données
features = ['Energy_kcal', 'Protein_g', 'Saturated_fats_g', 
            'Fat_g', 'Sugar_g', 'VitA_mcg', 'VitD2_mcg', 'VitE_mg']
X = df[features]
y = df['Target']

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Configuration des modèles
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Entraînement et évaluation
results = {}
for name, model in models.items():
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédiction
    y_pred = model.predict(X_test)
    
    # Stockage des résultats
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'cross_val': cross_val_score(model, X_scaled, y, cv=5).mean()
    }
    
    # Affichage des résultats
    print(f"\n=== {name} ===")
    print(f"Accuracy: {results[name]['accuracy']:.2%}")
    print(f"Cross-Validation Score: {results[name]['cross_val']:.2%}")
    print("Matrice de confusion:")
    print(results[name]['confusion_matrix'])
    
    # Visualisation de la matrice de confusion
    plt.figure(figsize=(6, 4))
    sns.heatmap(results[name]['confusion_matrix'], 
                annot=True, fmt='d', 
                cmap='Blues',
                xticklabels=['Bad', 'Good'],
                yticklabels=['Bad', 'Good'])
    plt.title(f'Matrice de Confusion - {name}')
    plt.show()

# Interface de prédiction avec suggestions d'amélioration
def predict_interface(model):
    while True:
        print("\nEntrez les valeurs nutritionnelles (ou 'q' pour quitter):")
        inputs = {}
        try:
            food_name = input("Nom de l'aliment: ")
            inputs['Energy_kcal'] = float(input("Énergie (kcal): "))
            inputs['Protein_g'] = float(input("Protéines (g): "))
            inputs['Saturated_fats_g'] = float(input("Graisses saturées (g): "))
            inputs['Fat_g'] = float(input("Graisses totales (g): "))
            inputs['Sugar_g'] = float(input("Sucres (g): "))
            inputs['VitA_mcg'] = float(input("Vitamine A (mcg): "))
            inputs['VitD2_mcg'] = float(input("Vitamine D (mcg): "))
            inputs['VitE_mg'] = float(input("Vitamine E (mg): "))
            
            new_data = pd.DataFrame([inputs])
            scaled_data = scaler.transform(new_data)
            
            prediction = model.predict(scaled_data)[0]
            probabilities = model.predict_proba(scaled_data)[0]
            
            print(f"\nRésultat pour '{food_name}': {prediction}")
            print(f"Probabilités: [Bad: {probabilities[0]:.2%}, Good: {probabilities[1]:.2%}]")
            
            # Identification des colonnes faibles pour un "Bad"
            if prediction == 'Bad':
                print("\nColonnes à améliorer pour passer à 'Good':")
                weak_points = []
                if inputs['Energy_kcal'] >= 500:
                    weak_points.append("Énergie (kcal) : Réduire en dessous de 500 kcal")
                if inputs['Fat_g'] >= 25:
                    weak_points.append("Graisses totales (g) : Réduire en dessous de 25 g")
                if inputs['Sugar_g'] >= 18:
                    weak_points.append("Sucres (g) : Réduire en dessous de 18 g")
                if inputs['Protein_g'] <= 10:
                    weak_points.append("Protéines (g) : Augmenter au-dessus de 10 g")
                if inputs['Saturated_fats_g'] >= 10:
                    weak_points.append("Graisses saturées (g) : Réduire en dessous de 10 g")
                if inputs['VitA_mcg'] < 900:
                    weak_points.append("Vitamine A (mcg) : Augmenter au-dessus de 900 mcg")
                if inputs['VitD2_mcg'] < 150:
                    weak_points.append("Vitamine D (mcg) : Augmenter au-dessus de 150 mcg")
                if inputs['VitE_mg'] < 150:
                    weak_points.append("Vitamine E (mg) : Augmenter au-dessus de 150 mg")
                
                if weak_points:
                    print("\nSuggestions d'amélioration :")
                    for point in weak_points:
                        print(f"- {point}")
                else:
                    print("Aucune amélioration nécessaire (erreur de prédiction).")
            
        except ValueError:
            if input("Valeur invalide. Voulez-vous quitter? (q/autre): ") == 'q':
                break
            continue
            
        if input("\nNouvelle prédiction? (o/n): ").lower() != 'o':
            break

# Choix du modèle
print("\nModèles disponibles:")
print(", ".join(models.keys()))
selected_model = input("Choisissez un modèle: ").strip()

if selected_model in models:
    predict_interface(models[selected_model])
else:
    print("Modèle non reconnu.")
    
# Assuming your trained model is stored in a variable called 'model'
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully as model.pkl")