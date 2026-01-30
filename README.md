## Équipe
- **Nom de l’institut :ISPM : Institut Supérieur Polytechnique de Madagascar  
- **Site web : [www.ispm-edu.com](http://www.ispm-edu.com)

## Membres de l’équipe
- ANDRIANJOHANY Liantsoa Nomban'Ny Avo, son rôle "développement backend" IGGLIA5
- ANDRIANANDRASANA Finiaina, son rôle "développement frontend" IGGLIA5
- RAKOTOMALALA Aina Anjaratiana, son rôle "Création dataset" IGGLIA5
- RAHARINAIVO Faramampionona, son rôle "déploiemment de ce projet" IGGLIA5

## Stack technologique
- Python, Flask, HTML/CSS
- Librairies ML : scikit-learn, pandas,numpy, Werkzeug, Jinja2, itsdangerous

## Description du processus et du modèle
Ce projet est une application de Machine Learning permettant de classer des messages

Le workflow est le suivant :  
1.Chargement des données: le fichier `dataset.csv` contient les messages et leurs étiquettes
2.Prétraitement des données: nettoyage du texte, suppression des caractères spéciaux, tokenization
3.Entraînement du modèle ML:  
   - Utilisation d’un modèle de classification (ex : `MultinomialNB` pour le spam) sur les données
   - Enregistrement du modèle entraîné dans `spam_model.pkl` et du vectorizer dans `vectorizer.pkl`.   
4.Interface utilisateur: l’application web (`interface.html` + `app.py`) permet de saisir un message
5.Workflow général:  
   - L’utilisateur saisit un texte → le texte est prétraité → le modèle prédit si c’est un spam ou non

## Méthodes ML utilisées

- Vectorisation du texte: `CountVectorizer` ou `TfidfVectorizer` pour transformer le texte en vecteurs>- Classification: `Multinomial Naive Bayes` pour prédire si un message est un spam ou non.  
- Évaluation du modèle: précision, rappel, F1-score (si tu fais des tests de validation).  

