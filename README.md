# Projet ML - Prédiction du Churn Telco

Application Streamlit interactive de prédiction du churn (résiliation client) pour un opérateur télécom, avec pipeline ML complet : exploration, préparation, modélisation et évaluation.

## Fonctionnalités

- **Exploration des données** : statistiques descriptives, distributions, boxplots, matrice de corrélation
- **Analyse bivariée** : facteurs de risque clés par segment (contrat, service internet, satisfaction)
- **Pipeline ML** : préprocessing (StandardScaler, OneHotEncoder, TF-IDF) + SMOTE + RandomForest
- **Évaluation interactive** : slider de seuil de décision, métriques Train/Test, ROC, matrice de confusion
- **Courbe d'apprentissage** : détection overfitting / underfitting
- **Analyse textuelle** : WordClouds churners vs non-churners (FeedbackText)
- **Segments à risque** : barplots horizontaux par feature avec alertes configurables

## Structure du projet

```
projet_ml_Telcox/
├── app.py                  # Application Streamlit principale
├── ml_project.ipynb        # Notebook d'exploration et expérimentation
├── Rapport ml projet.pdf   # Rapport complet du projet
├── requirements.txt        # Dépendances Python
├── .gitignore
└── data/
    ├── clients.csv         # Données clients (5 467 lignes)
    ├── contracts.csv       # Données contrats
    ├── interactions.csv    # Données interactions support
    └── usage.csv           # Données usage réseau
```

## Données

Les 4 fichiers CSV sont fusionnés sur `customerID` :

| Fichier | Colonnes principales |
|---|---|
| `clients.csv` | customerID, gender, SeniorCitizen, Partner, Dependents, tenure, Age, **Churn** |
| `contracts.csv` | ContractType, MonthlyCharges, TotalCharges, PaymentMethod, InternetService |
| `interactions.csv` | NbContacts, LastContactDays, SatisfactionScore, FeedbackText |
| `usage.csv` | AvgDataUsage_GB, NumCalls, TVPackage, TechSupport |

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Nour-1990/projet_ml_Telcox.git
cd projet_ml_Telcox

# 2. Créer et activer l'environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à `http://localhost:8501`.

## Modèle ML

- **Algorithme** : RandomForestClassifier
- **Rééchantillonnage** : SMOTE (gestion du déséquilibre de classes)
- **Texte** : TF-IDF sur la colonne `FeedbackText` (200 features max)
- **Paramètres** : `n_estimators=200`, `max_depth=10`, `class_weight=balanced_subsample`

## Auteur

Zerabib Nour — Projet ML Telco Churn Prediction
