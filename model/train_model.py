import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# -----------------------
# Simulate realistic data
# -----------------------
np.random.seed(42)
n = 2000

# Age: normal distribution 18–80
age = np.random.randint(18, 80, n)

# Vitamin D: correlation with ferritin (low vit D often -> low ferritin)
vitamin_d = np.random.normal(40, 15, n).clip(5, 100)
ferritin = np.where(vitamin_d < 20,
                    np.random.normal(15, 5, n),
                    np.random.normal(80, 30, n)).clip(5, 300)

# A1C: correlate with glucose
a1c = np.random.normal(5.5, 0.6, n).clip(4.5, 9)
glucose = np.where(a1c > 6,
                   np.random.normal(130, 20, n),
                   np.random.normal(95, 15, n)).clip(60, 250)

# Creatinine: mildly correlated with age
creatinine = np.random.normal(0.8 + (age / 100) * 0.3, 0.15, n).clip(0.4, 2.5)

# LDL: correlate with age and glucose
ldl = np.random.normal(100 + (age / 2) + (glucose / 50), 25, n).clip(50, 250)

# Vitamin B12: random distribution
vitamin_b12 = np.random.normal(500, 150, n).clip(100, 1100)

# TSH: 10% abnormal
tsh = np.where(
    np.random.rand(n) < 0.1,
    np.where(np.random.rand(n) < 0.5, np.random.uniform(0.01, 0.29, n), np.random.uniform(4.6, 10, n)),
    np.random.uniform(0.3, 4.5, n)
)

# -----------------------
# Create DataFrame
# -----------------------
df = pd.DataFrame({
    "age": age,
    "vitamin_d": vitamin_d,
    "a1c": a1c,
    "ferritin": ferritin,
    "glucose": glucose,
    "creatinine": creatinine,
    "ldl": ldl,
    "vitamin_b12": vitamin_b12,
    "tsh": tsh
})

# -----------------------
# Target labels
# -----------------------
df["vitamin_d_deficiency"] = df["vitamin_d"] < 20
df["anemia_risk"] = (df["ferritin"] < 25) & (df["vitamin_d"] < 30)
df["prediabetes_risk"] = (df["a1c"] >= 5.7) & (df["a1c"] < 6.5)
df["thyroid_flag"] = (df["tsh"] < 0.3) | (df["tsh"] > 4.5)
df["high_cholesterol_risk"] = (df["ldl"] > 130) | ((df["glucose"] > 120) & (df["age"] > 50))

# -----------------------
# Train/Test Split
# -----------------------
X = df[["age", "vitamin_d", "a1c", "ferritin", "glucose", "creatinine", "ldl", "vitamin_b12", "tsh"]]
y = df[["vitamin_d_deficiency", "anemia_risk", "prediabetes_risk", "thyroid_flag", "high_cholesterol_risk"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# Train model
# -----------------------
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=300, random_state=42))
clf.fit(X_train, y_train)

# -----------------------
# Save model
# -----------------------
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/illness_risk_model.joblib")

# -----------------------
# Evaluation
# -----------------------
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

