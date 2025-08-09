from flask import Flask, render_template, request
from model.predict import predict_illness_scores, FEATURE_ORDER
from pdf_parser.extract_lab_values import extract_lab_features
import tempfile
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files.get('pdf')

        # Validate file presence and extension
        if not uploaded_file or not uploaded_file.filename.lower().endswith('.pdf'):
            return render_template('upload.html', error="Please upload a valid PDF file.")

        # Save uploaded PDF temporarily to process
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            uploaded_file.save(tmp.name)
            pdf_path = tmp.name

        # Extract lab features
        try:
            lab_values = extract_lab_features(pdf_path)
            print("DEBUG lab_values ->", lab_values)
        except Exception:
            return render_template('upload.html', error="Error reading the PDF file.")
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)  # cleanup temp file

        # Identify missing values for strict mode
        missing_values = [feat for feat in FEATURE_ORDER if lab_values.get(feat) is None]

        # Run prediction only if all values present
        if missing_values:
            results = predict_illness_scores(lab_values)  # still run prediction with available data
            return render_template(
                'result.html',
                results=results,
                missing_values=missing_values
            )

        # No missing values: run full prediction
        results = predict_illness_scores(lab_values)
        return render_template('result.html', results=results, missing_values=[])

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)



