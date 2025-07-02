#!/usr/bin/env python3
"""
Flask API for Marine Accident Severity Classification
Deploy this to connect your web interface to real ML models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for models
model = None
vectorizer = None
severity_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

def load_models():
    """Load the trained models"""
    global model, vectorizer
    
    try:
        # Try to load the improved model first
        model_path = 'models/improved_marine_accident_classifier_model.pkl'
        vectorizer_path = 'models/improved_marine_accident_classifier_vectorizer.pkl'
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            logger.info("Loading improved marine accident classifier...")
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            logger.info("✅ Models loaded successfully!")
            return True
        else:
            # Try simple model as fallback
            model_path = 'models/marine_accident_classifier_model.pkl'
            vectorizer_path = 'models/marine_accident_classifier_vectorizer.pkl'
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                logger.info("Loading simple marine accident classifier...")
                model = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
                logger.info("✅ Models loaded successfully!")
                return True
            else:
                logger.error("❌ No trained models found!")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return False

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Marine Accident Severity Classification API',
        'version': '1.0.0',
        'endpoints': {
            '/classify': 'POST - Classify accident severity',
            '/health': 'GET - Check API health',
            '/models': 'GET - Get model information'
        },
        'usage': {
            'method': 'POST',
            'url': '/classify',
            'body': {'description': 'Accident description text'}
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    models_loaded = model is not None and vectorizer is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'timestamp': str(np.datetime64('now'))
    })

@app.route('/models')
def model_info():
    """Get information about loaded models"""
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Models not loaded',
            'status': 'unavailable'
        }), 503
    
    return jsonify({
        'model_type': type(model).__name__,
        'vectorizer_type': type(vectorizer).__name__,
        'severity_classes': list(severity_mapping.values()),
        'status': 'loaded'
    })

@app.route('/classify', methods=['POST'])
def classify():
    """Classify accident severity"""
    try:
        # Check if models are loaded
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Models not loaded. Please check /health endpoint.',
                'status': 'error'
            }), 503
        
        # Get request data
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({
                'error': 'Missing description field',
                'status': 'error'
            }), 400
        
        description = data['description'].strip()
        
        if not description:
            return jsonify({
                'error': 'Description cannot be empty',
                'status': 'error'
            }), 400
        
        # Log the request
        logger.info(f"Classifying: {description[:100]}...")
        
        # Transform text using TF-IDF
        text_tfidf = vectorizer.transform([description])
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        
        # Map prediction to severity
        severity = severity_mapping[prediction]
        confidence = float(max(probabilities))
        
        # Create response
        result = {
            'severity': severity,
            'confidence': confidence,
            'probabilities': {
                'low': float(probabilities[0]),
                'medium': float(probabilities[1]),
                'high': float(probabilities[2])
            },
            'prediction_numeric': int(prediction),
            'status': 'success'
        }
        
        logger.info(f"Prediction: {severity} (confidence: {confidence:.3f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        return jsonify({
            'error': 'Internal server error during classification',
            'status': 'error'
        }), 500

@app.route('/classify', methods=['GET'])
def classify_get():
    """GET endpoint for testing with query parameter"""
    description = request.args.get('description', '')
    
    if not description:
        return jsonify({
            'error': 'Missing description parameter',
            'usage': 'GET /classify?description=your_accident_description'
        }), 400
    
    # Create a mock POST request
    mock_data = {'description': description}
    request._json = mock_data
    return classify()

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/health', '/models', '/classify']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

def main():
    """Main function to run the Flask app"""
    # Load models on startup
    if not load_models():
        logger.warning("⚠️  Starting without models. Use /health to check status.")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"🚀 Starting Marine Accident Classification API on port {port}")
    logger.info(f"📊 API Documentation: http://localhost:{port}/")
    logger.info(f"🏥 Health Check: http://localhost:{port}/health")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

if __name__ == '__main__':
    main() 