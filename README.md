# Health Risk Predictor
https://health-risk-predictor-0te6.onrender.com

Health Risk Predictor is a Python-based web application that analyzes laboratory results from PDF files and predicts potential health risks using a trained machine learning model. It also highlights the top contributing lab factors for each predicted risk.

---

##  Features
- **PDF Parsing** — Extracts lab values from uploaded PDF reports using `pdfplumber`.
- **Data Preprocessing** — Cleans and formats extracted data for prediction.
- **Machine Learning Predictions** — Uses a `RandomForestClassifier` to assess multiple health risks simultaneously.
- **Top Risk Factors** — Shows which lab results contributed most to the prediction.
- **Web Interface** — Built with Flask for easy interaction.

---

##  Tech Stack
- **Backend:** Python 3, Flask  
- **Machine Learning:** scikit-learn, pandas, numpy  
- **PDF Parsing:** pdfplumber  
- **Model Storage:** joblib  
- **Visualization / Styling:** HTML, CSS (Tailwind CSS in templates)

##  Model Training

This project uses a **synthetic dataset** that mimics realistic patterns found in common lab results.  
I generated **2,000 virtual patient records** using **NumPy** and **Pandas**, ensuring correlations between biomarkers that are typically seen in medical contexts.

---

###  Data Simulation Steps
- **Age:** Random integers between 18–80.  
- **Vitamin D & Ferritin:** Correlated — low Vitamin D often implies low ferritin.  
- **A1C & Glucose:** Higher A1C leads to higher glucose values.  
- **Creatinine:** Slightly correlated with age.  
- **LDL:** Correlated with both age and glucose.  
- **Vitamin B12:** Random distribution.  
- **TSH:** About 10% abnormal values to simulate thyroid issues.  

---

###  Risk Labels Generated
- **vitamin_d_deficiency** → Vitamin D < 20 ng/mL  
- **anemia_risk** → Low ferritin with low Vitamin D  
- **prediabetes_risk** → A1C between 5.7 and 6.5  
- **thyroid_flag** → TSH < 0.3 or > 4.5  
- **high_cholesterol_risk** → LDL > 130 or elevated glucose with age > 50  

---

###  Model
- **Algorithm:** RandomForestClassifier wrapped in MultiOutputClassifier (predicts all risks at once).  
- **Training/Test Split:** 80/20.  
- **Model Export:** Saved with `joblib` to `model/illness_risk_model.joblib`.

###  Example Prediction Flow
1. **Upload Lab Report (PDF)** – User uploads a PDF lab report.  
2. **System Extracts Data** – Uses `pdfplumber` to find and extract relevant lab values.  
3. **Prediction Engine** – Processes the extracted values through the trained machine learning model.  
4. **Result Display** – Shows predicted health risks and their top contributing lab factors.  

---

###  Disclaimer
This application is for **demonstration purposes only**.
