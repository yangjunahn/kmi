import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class EnhancedMarineAccidentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.severity_mapping = {'low': 0, 'medium': 1, 'high': 2}
        self.reverse_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        
    def load_and_train(self, csv_file):
        """Load data and train the model"""
        print("Loading and training enhanced marine accident classifier...")
        
        # Load data
        df = pd.read_csv(csv_file)
        df_clean = df.dropna(subset=['Description', 'Severity'])
        df_clean['severity_numeric'] = df_clean['Severity'].map(self.severity_mapping)
        
        # Prepare data
        X = df_clean['Description']
        y = df_clean['severity_numeric']
        
        # Enhanced TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 4),  # Include 4-grams for better phrase capture
            min_df=1,
            max_df=0.95,
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Compute class weights with higher penalty for minority classes
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        # Increase weights for medium and high severity
        class_weights[1] *= 1.5  # Medium severity
        class_weights[2] *= 2.0  # High severity
        weight_dict = dict(zip(np.unique(y), class_weights))
        
        print(f"Class weights: {weight_dict}")
        
        # Train model with more trees and deeper
        self.model = RandomForestClassifier(
            n_estimators=500, 
            random_state=42,
            class_weight=weight_dict,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1
        )
        self.model.fit(X_tfidf, y)
        
        print(f"Model trained on {len(X)} accident records")
        print("Ready for predictions!")
    
    def predict_accident(self, description):
        """Predict severity for an accident description"""
        if self.model is None:
            return "Model not trained yet!"
        
        # Transform text
        text_tfidf = self.vectorizer.transform([description])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Map back to severity labels
        severity = self.reverse_mapping[prediction]
        confidence = max(probabilities)
        
        # Generate reasoning
        reasoning = self.generate_reasoning(description, severity, probabilities)
        
        return {
            'severity': severity,
            'confidence': confidence,
            'probabilities': probabilities,
            'reasoning': reasoning
        }
    
    def generate_reasoning(self, text, severity, probabilities):
        """Generate detailed reasoning for the prediction"""
        # Enhanced severity indicators with more specific terms
        high_severity_words = [
            'collision', 'fire', 'explosion', 'sinking', 'casualty', 'death', 
            'serious injury', 'major damage', 'oil spill', 'evacuation', 'multiple',
            'fatal', 'critical', 'emergency', 'disaster', 'catastrophic', 'sunk',
            'capsized', 'abandoned', 'rescue', 'hospital', 'medical', 'missing',
            'sink', 'sinking', 'burning', 'burned', 'exploded', 'explosion',
            'casualties', 'injuries', 'fatalities', 'dead', 'killed', 'lost',
            'major collision', 'severe damage', 'extensive damage', 'total loss'
        ]
        
        medium_severity_words = [
            'grounding', 'allision', 'minor damage', 'delay', 'operational issue',
            'structural damage', 'navigation error', 'equipment failure', 'contact',
            'stuck', 'aground', 'stranded', 'leak', 'flooding', 'engine failure',
            'steering failure', 'electrical failure', 'mechanical failure',
            'grounded', 'ran aground', 'stuck in', 'damaged', 'damage to',
            'operational delay', 'temporary', 'partial', 'limited'
        ]
        
        low_severity_words = [
            'minor', 'slight', 'no damage', 'operational', 'routine', 'maintenance',
            'weather', 'technical issue', 'delay', 'temporary', 'repair', 'inspection',
            'preventive', 'scheduled', 'normal', 'regular', 'standard', 'minor issue',
            'small', 'light', 'minimal', 'insignificant', 'routine maintenance'
        ]
        
        text_lower = text.lower()
        high_count = sum(1 for word in high_severity_words if word in text_lower)
        medium_count = sum(1 for word in medium_severity_words if word in text_lower)
        low_count = sum(1 for word in low_severity_words if word in text_lower)
        
        confidence = max(probabilities)
        
        # Enhanced reasoning with more context
        if severity == 'high':
            if high_count > 0:
                found = [w for w in high_severity_words if w in text_lower]
                return f"High severity indicators detected: '{', '.join(found)}'. This suggests serious consequences including potential loss of life, major property damage, environmental impact, or significant operational disruptions requiring emergency response."
            else:
                return "The model's analysis suggests this incident has serious consequences that could impact safety, environment, or major operational disruptions, despite not containing explicit high-severity keywords."
        elif severity == 'medium':
            if medium_count > 0:
                found = [w for w in medium_severity_words if w in text_lower]
                return f"Medium severity indicators detected: '{', '.join(found)}'. This suggests moderate consequences with some operational impact but manageable safety risks and standard response procedures."
            else:
                return "The model's analysis suggests moderate consequences with some operational impact but manageable safety and environmental risks."
        else:  # low
            if low_count > 0:
                found = [w for w in low_severity_words if w in text_lower]
                return f"Low severity indicators detected: '{', '.join(found)}'. This suggests minor consequences with minimal operational impact and routine procedures."
            else:
                return "The model's analysis suggests minor consequences with minimal operational impact and low safety risks."
    
    def analyze_text_features(self, text):
        """Analyze the text for severity indicators"""
        text_lower = text.lower()
        
        # Count different types of severity indicators
        high_words = ['collision', 'fire', 'explosion', 'sinking', 'casualty', 'death', 'major damage', 'oil spill', 'evacuation', 'missing', 'sunk', 'burning', 'exploded', 'fatalities', 'dead', 'killed', 'lost', 'severe damage', 'extensive damage', 'total loss']
        medium_words = ['grounding', 'allision', 'minor damage', 'delay', 'structural damage', 'navigation error', 'equipment failure', 'contact', 'stuck', 'aground', 'stranded', 'leak', 'flooding', 'engine failure', 'grounded', 'ran aground', 'damaged', 'operational delay', 'temporary', 'partial']
        low_words = ['minor', 'slight', 'no damage', 'operational', 'routine', 'maintenance', 'weather', 'technical issue', 'delay', 'temporary', 'repair', 'inspection', 'preventive', 'scheduled', 'normal', 'regular', 'standard', 'minor issue', 'small', 'light', 'minimal']
        
        high_count = sum(1 for word in high_words if word in text_lower)
        medium_count = sum(1 for word in medium_words if word in text_lower)
        low_count = sum(1 for word in low_words if word in text_lower)
        
        return {
            'high_indicators': high_count,
            'medium_indicators': medium_count,
            'low_indicators': low_count,
            'total_indicators': high_count + medium_count + low_count
        }

def test_enhanced_classifier():
    """Test the enhanced marine accident classifier"""
    
    # Initialize classifier
    classifier = EnhancedMarineAccidentClassifier()
    
    # Train model
    classifier.load_and_train('marine_accident_reports.csv')
    
    # Test cases with expected severity levels
    test_cases = [
        {
            'description': "Collision between two cargo ships in the harbor resulting in major hull damage and oil spill",
            'expected': 'high'
        },
        {
            'description': "Minor grounding incident with no damage to vessel or environment",
            'expected': 'low'
        },
        {
            'description': "Fire outbreak in engine room causing evacuation of crew members",
            'expected': 'high'
        },
        {
            'description': "Allision with dock structure causing minor damage to bow",
            'expected': 'medium'
        },
        {
            'description': "Explosion in cargo hold leading to multiple casualties",
            'expected': 'high'
        },
        {
            'description': "Navigation error causing vessel to run aground in shallow water",
            'expected': 'medium'
        },
        {
            'description': "Routine maintenance issue causing temporary operational delay",
            'expected': 'low'
        },
        {
            'description': "Weather-related incident causing minor structural damage",
            'expected': 'low'
        },
        {
            'description': "Major collision with bridge structure causing vessel to sink",
            'expected': 'high'
        },
        {
            'description': "Engine room fire resulting in complete power loss and emergency response",
            'expected': 'high'
        },
        {
            'description': "There was a collision accident in front of the Busan port. One man is missing.",
            'expected': 'high'
        },
        {
            'description': "Minor technical issue during routine inspection",
            'expected': 'low'
        },
        {
            'description': "Vessel ran aground due to navigation error in foggy conditions",
            'expected': 'medium'
        },
        {
            'description': "Fire broke out in the cargo hold during loading operations",
            'expected': 'high'
        },
        {
            'description': "Structural damage to hull during severe weather conditions",
            'expected': 'medium'
        }
    ]
    
    print("\n" + "="*100)
    print("ENHANCED MARINE ACCIDENT SEVERITY CLASSIFICATION TEST RESULTS")
    print("="*100)
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        accident = test_case['description']
        expected = test_case['expected']
        
        print(f"\n{i}. ACCIDENT DESCRIPTION:")
        print(f"   {accident}")
        print(f"   Expected Severity: {expected.upper()}")
        
        try:
            result = classifier.predict_accident(accident)
            text_analysis = classifier.analyze_text_features(accident)
            
            print(f"\n   PREDICTION: {result['severity'].upper()}")
            print(f"   CONFIDENCE: {result['confidence']:.1%}")
            print(f"   PROBABILITIES: Low: {result['probabilities'][0]:.1%}, Medium: {result['probabilities'][1]:.1%}, High: {result['probabilities'][2]:.1%}")
            print(f"   REASONING: {result['reasoning']}")
            
            # Text analysis
            print(f"   TEXT ANALYSIS: High indicators: {text_analysis['high_indicators']}, Medium: {text_analysis['medium_indicators']}, Low: {text_analysis['low_indicators']}")
            
            # Check if prediction matches expected
            if result['severity'] == expected:
                correct_predictions += 1
                print(f"   ✓ CORRECT PREDICTION")
            else:
                print(f"   ✗ INCORRECT PREDICTION (Expected: {expected.upper()})")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        print("-" * 100)
    
    accuracy = correct_predictions / total_predictions
    print(f"\nFINAL RESULTS:")
    print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.1%}")
    print("="*100)

if __name__ == "__main__":
    test_enhanced_classifier() 