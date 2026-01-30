from flask import Flask, render_template, request
import joblib
import re
import os

app = Flask(__name__)

# Chargement du modèle et du vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', texte)
    return texte


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confiance = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        texte = nettoyer_texte(message)

        vect = vectorizer.transform([texte])
        proba = model.predict_proba(vect)[0]

        if proba[1] > proba[0]:
            prediction = "SPAM"
            confiance = round(proba[1] * 100, 2)
        else:
            prediction = "HAM"
            confiance = round(proba[0] * 100, 2)

    return render_template(
        "interface.html",
        prediction=prediction,
        confiance=confiance,
        message=message
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render fournit le port ici
    app.run(host="0.0.0.0", port=port, debug=True)
