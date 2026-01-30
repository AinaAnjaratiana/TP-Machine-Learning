import os
import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Chargement du dataset
# Le fichier doit contenir : label, message
data = pd.read_csv("dataset.csv", encoding="latin-1")
data = data[['label', 'message']]

# Conversion label en numérique
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 2. Nettoyage du texte
def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', texte)
    return texte

data['message'] = data['message'].apply(nettoyer_texte)

# 3. Séparation train / test
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 4. Vectorisation TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Modèle Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Évaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy du modèle : {accuracy * 100:.2f}%")

# 7. Création du dossier model/ s'il n'existe pas
if not os.path.exists("model"):
    os.makedirs("model")

# 8. Sauvegarde
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Modèle et vectorizer sauvegardés avec succès")
