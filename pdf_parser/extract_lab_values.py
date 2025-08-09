import pdfplumber
import re

def extract_lab_features(pdf_path):
    features = {
        "age": None,
        "sex": None,
        "vitamin_d": None,
        "a1c": None,
        "ferritin": None,
        "glucose": None,
        "creatinine": None,
        "ldl": None,
        "vitamin_b12": None,
        "tsh": None,
    }

    # Read all text from the PDF
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )

    # Patterns to extract the features
    patterns = {
        "age": r"Age:\s*(\d+)",
        "sex": r"Sex:\s*([FM])",
        "vitamin_d": r"VITAMIN D,25-OH,TOTAL,IA\s+([\d.]+)",
        "a1c": r"HEMOGLOBIN A1c\s+([\d.]+)",
        "ferritin": r"FERRITIN\s+([\d.]+)",
        "glucose": r"GLUCOSE\s+([\d.]+)",
        "creatinine": r"CREATININE\s+([\d.]+)",
        "ldl": r"LDL-CHOLESTEROL\s+([\d.]+)",
        "vitamin_b12": r"VITAMIN B12\s+([\d.]+)",
        "tsh": r"TSH\s+([\d.]+)",
    }

    # Apply regex to extract each value
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if key == "sex":
                features[key] = value
            else:
                try:
                    features[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    features[key] = None
        else:
            # Strict mode: leave as None if not found
            features[key] = None

    return features

