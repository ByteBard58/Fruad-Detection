'''
app.py

This is the main application file for the fraud detection system.
It handles the web interface and file processing.
'''

import os
import secrets
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_file, redirect

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
MODEL_PATH = "models/pipe.pkl"
FEAT_NAMES_PATH = "models/feat_names.pkl"
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Global model variables
pipeline = None
feature_names = None

def load_model():
    global pipeline, feature_names
    if os.path.exists(MODEL_PATH) and os.path.exists(FEAT_NAMES_PATH):
        try:
            print(f"Loading model from {MODEL_PATH}...")
            pipeline = joblib.load(MODEL_PATH)
            feature_names = joblib.load(FEAT_NAMES_PATH)
            print("Model and features loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            pipeline = None
            feature_names = None
    else:
        print("Model files not found. Please run fit.py first.")

# Load model on startup
load_model()


# Import necessary for joblib to load the pipeline correctly if it uses imblearn
try:
    from imblearn.pipeline import Pipeline
except ImportError:
    from sklearn.pipeline import Pipeline

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for automated verification."""
    status = "ok" if pipeline is not None else "degraded (model missing)"
    return jsonify({"status": status}), 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if pipeline is None:
        return render_template('index.html', error="Model is not loaded. Please contact administrator.")

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secrets.token_hex(8) + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Basic validation
            # We assume the user uploads a file with the same columns as training, 
            # possibly including the target 'Class' or 'id' which we might need to drop.
            # fit.py drops 'id' and usage all columns except last as features.
            # Here, we need to match feature_names.
            
            # Align columns
            # 1. Drop 'id' if exists
            if 'id' in df.columns:
                df_for_pred = df.drop(columns=['id'])
            else:
                df_for_pred = df.copy()
            
            # 2. Drop target if exists (e.g. 'Class') - usually not in 'new' data but if testing with raw data
            if 'Class' in df_for_pred.columns:
                df_for_pred = df_for_pred.drop(columns=['Class'])

            # 3. Ensure all features exist
            missing_cols = set(feature_names) - set(df_for_pred.columns)
            if missing_cols:
                return render_template('index.html', error=f"Missing columns: {missing_cols}")
            
            # 4. Reorder to match training
            df_for_pred = df_for_pred[feature_names]

            # Predict
            # 1 is Fraud, 0 is Clean
            preds = pipeline.predict(df_for_pred)
            probs = pipeline.predict_proba(df_for_pred)[:, 1] # Probability of class 1 (Fraud)

            # Attach results
            df['Fraud_Prediction'] = preds
            df['Fraud_Probability'] = probs

            # Calculate Stats
            total_tx = len(df)
            fraud_tx = int(sum(preds))
            clean_tx = total_tx - fraud_tx
            fraud_pct = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0

            # Get Top 10 Highest Risk
            # Sort by probability descending
            # We convert to records for Jinja
            top_risky = df.sort_values(by='Fraud_Probability', ascending=False).head(10).to_dict(orient='records')
            
            # Save processed
            processed_filename = "analyzed_" + filename
            processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
            df.to_csv(processed_path, index=False)

            return render_template(
                'dashboard.html',
                total=total_tx,
                fraud=fraud_tx,
                clean=clean_tx,
                fraud_pct=f"{fraud_pct:.2f}",
                top_risky=top_risky,
                download_link=processed_filename
            )

        except Exception as e:
            return render_template('index.html', error=f"Error processing file: {str(e)}")

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
