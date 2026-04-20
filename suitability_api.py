"""
CROP SUITABILITY CLASSIFICATION API
Deploy this as a separate service on Render
"""

import numpy as np
import pandas as pd
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("CROP SUITABILITY API - Starting up...")
print("=" * 60)

# ============================================
# LOAD MODELS
# ============================================

# Rice models
rice_model_path = os.path.join(BASE_DIR, 'rice_suitability_model.pkl')
rice_scaler_path = os.path.join(BASE_DIR, 'rice_scaler.pkl')
rice_encoder_path = os.path.join(BASE_DIR, 'rice_label_encoder.pkl')
rice_features_path = os.path.join(BASE_DIR, 'rice_features.npy')

# Corn models
corn_model_path = os.path.join(BASE_DIR, 'corn_suitability_model.pkl')
corn_scaler_path = os.path.join(BASE_DIR, 'corn_scaler.pkl')
corn_encoder_path = os.path.join(BASE_DIR, 'corn_label_encoder.pkl')
corn_features_path = os.path.join(BASE_DIR, 'corn_features.npy')

# Load models
rice_model = joblib.load(rice_model_path) if os.path.exists(rice_model_path) else None
corn_model = joblib.load(corn_model_path) if os.path.exists(corn_model_path) else None

# Load scalers
rice_scaler = joblib.load(rice_scaler_path) if os.path.exists(rice_scaler_path) else None
corn_scaler = joblib.load(corn_scaler_path) if os.path.exists(corn_scaler_path) else None

# Load label encoders
rice_encoder = joblib.load(rice_encoder_path) if os.path.exists(rice_encoder_path) else None
corn_encoder = joblib.load(corn_encoder_path) if os.path.exists(corn_encoder_path) else None

# Load feature lists
rice_features = np.load(rice_features_path, allow_pickle=True) if os.path.exists(rice_features_path) else None
corn_features = np.load(corn_features_path, allow_pickle=True) if os.path.exists(corn_features_path) else None

print(f"✅ Rice Model loaded: {rice_model is not None}")
print(f"✅ Corn Model loaded: {corn_model is not None}")
print(f"✅ Rice Scaler loaded: {rice_scaler is not None}")
print(f"✅ Corn Scaler loaded: {corn_scaler is not None}")
print("=" * 60)

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_derived_features(ndvi, evi, temperature, rainfall, soil_fertility, n_score, p_score, k_score):
    """Calculate derived features for prediction"""
    veg_health = ndvi * evi
    gdd = max(0, temperature - 10)
    
    features = {
        'ndvi': ndvi,
        'evi': evi,
        'temperature': temperature,
        'rainfall_total': rainfall,
        'humidity': 75,
        'gdd': gdd,
        'veg_health': veg_health,
        'soil_fertility': soil_fertility,
        'ndvi_evi_product': ndvi * evi,
        'soil_veg': soil_fertility * veg_health,
        'temp_rain': temperature * rainfall / 1000,
        'n_score': n_score,
        'p_score': p_score,
        'k_score': k_score,
        'ndvi_squared': ndvi ** 2,
        'evi_squared': evi ** 2,
        'ndvi_evi_ratio': ndvi / (evi + 0.001),
        'water_use_efficiency': veg_health / (rainfall + 0.001)
    }
    return features

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
        
        crop_type = data.get('crop', '').lower()
        ndvi = data.get('ndvi')
        evi = data.get('evi')
        temperature = data.get('temperature')
        rainfall = data.get('rainfall')
        soil_fertility = data.get('soil_fertility')
        n_score = data.get('n_score', 3)
        p_score = data.get('p_score', 3)
        k_score = data.get('k_score', 3)
        humidity = data.get('humidity', 75)
        
        # Validate inputs
        if crop_type not in ['rice', 'corn']:
            return jsonify({'error': 'Crop must be "rice" or "corn"'}), 400
        
        if any(v is None for v in [ndvi, evi, temperature, rainfall, soil_fertility]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Select appropriate model
        if crop_type == 'rice':
            model = rice_model
            scaler = rice_scaler
            encoder = rice_encoder
            feature_list = rice_features
        else:
            model = corn_model
            scaler = corn_scaler
            encoder = corn_encoder
            feature_list = corn_features
        
        if model is None:
            return jsonify({'error': f'{crop_type.capitalize()} model not loaded'}), 500
        
        print(f"\n{'='*60}")
        print(f"📥 SUITABILITY REQUEST: {crop_type.capitalize()}")
        print(f"{'='*60}")
        print(f"   NDVI: {ndvi}")
        print(f"   EVI: {evi}")
        print(f"   Temperature: {temperature}°C")
        print(f"   Rainfall: {rainfall} mm")
        print(f"   Soil Fertility: {soil_fertility}")
        
        # Calculate features
        features_dict = calculate_derived_features(
            ndvi, evi, temperature, rainfall, soil_fertility, n_score, p_score, k_score
        )
        features_dict['humidity'] = humidity
        
        # Create feature vector in correct order
        feature_vector = []
        for f in feature_list:
            feature_vector.append(features_dict.get(f, 0))
        
        # Scale features
        X_scaled = scaler.transform([feature_vector])
        
        # Predict
        pred_encoded = model.predict(X_scaled)[0]
        suitability = encoder.inverse_transform([pred_encoded])[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)[0]
            confidence = max(probs) * 100
            probabilities = {cls: prob * 100 for cls, prob in zip(encoder.classes_, probs)}
        else:
            confidence = None
            probabilities = None
        
        print(f"\n✅ SUITABILITY: {suitability}")
        if confidence:
            print(f"   Confidence: {confidence:.1f}%")
        print(f"{'='*60}\n")
        
        response = {
            'crop': crop_type.capitalize(),
            'suitability': suitability,
            'status': 'success'
        }
        
        if confidence:
            response['confidence'] = round(confidence, 1)
            response['probabilities'] = {k: round(v, 1) for k, v in probabilities.items()}
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'rice_model_loaded': rice_model is not None,
        'corn_model_loaded': corn_model is not None,
        'features_rice': rice_features.tolist() if rice_features is not None else None,
        'features_corn': corn_features.tolist() if corn_features is not None else None,
        'classes_rice': rice_encoder.classes_.tolist() if rice_encoder is not None else None,
        'classes_corn': corn_encoder.classes_.tolist() if corn_encoder is not None else None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
