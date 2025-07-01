import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class MarineAccidentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.severity_mapping = {'low': 0, 'medium': 1, 'high': 2}
        self.reverse_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        
    def load_and_train(self, csv_file):
        """Load data and train the model"""
        print("Loading and training marine accident classifier...")
        
        # Load data
        df = pd.read_csv(csv_file)
        df_clean = df.dropna(subset=['Description', 'Severity'])
        df_clean['severity_numeric'] = df_clean['Severity'].map(self.severity_mapping)
        
        # Prepare data
        X = df_clean['Description']
        y = df_clean['severity_numeric']
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9
        )
        
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        weight_dict = dict(zip(np.unique(y), class_weights))
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42,
            class_weight=weight_dict
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
        """Generate reasoning for the prediction"""
        # Severity indicators
        high_words = ['collision', 'fire', 'explosion', 'sinking', 'casualty', 'death', 'major damage', 'oil spill', 'evacuation']
        medium_words = ['grounding', 'allision', 'minor damage', 'delay', 'structural damage', 'navigation error']
        low_words = ['minor', 'slight', 'no damage', 'operational', 'routine', 'maintenance', 'weather']
        
        text_lower = text.lower()
        high_count = sum(1 for word in high_words if word in text_lower)
        medium_count = sum(1 for word in medium_words if word in text_lower)
        low_count = sum(1 for word in low_words if word in text_lower)
        
        confidence = max(probabilities)
        
        if severity == 'high':
            if high_count > 0:
                found = [w for w in high_words if w in text_lower]
                return f"High severity indicators found: {', '.join(found)}. Suggests serious consequences."
            else:
                return "Context suggests serious consequences requiring emergency response."
        elif severity == 'medium':
            if medium_count > 0:
                found = [w for w in medium_words if w in text_lower]
                return f"Medium severity indicators found: {', '.join(found)}. Suggests moderate consequences."
            else:
                return "Context suggests moderate consequences with manageable impact."
        else:
            if low_count > 0:
                found = [w for w in low_words if w in text_lower]
                return f"Low severity indicators found: {', '.join(found)}. Suggests minor consequences."
            else:
                return "Context suggests minor consequences with minimal impact."

def test_classifier():
    """Test the marine accident classifier with various scenarios"""
    
    # Initialize classifier
    classifier = MarineAccidentClassifier()
    
    # Train model
    classifier.load_and_train('marine_accident_reports.csv')
    
    # Test cases
    test_cases = [
        "Collision between two cargo ships in the harbor resulting in major hull damage and oil spill",
        "Minor grounding incident with no damage to vessel or environment",
        "Fire outbreak in engine room causing evacuation of crew members",
        "Allision with dock structure causing minor damage to bow",
        "Explosion in cargo hold leading to multiple casualties",
        "Navigation error causing vessel to run aground in shallow water",
        "Routine maintenance issue causing temporary operational delay",
        "Weather-related incident causing minor structural damage",
        "Major collision with bridge structure causing vessel to sink",
        "Engine room fire resulting in complete power loss and emergency response",
        "There was a collision accident in front of the Busan port. One man is missing.",
        "Minor technical issue during routine inspection",
        "Vessel ran aground due to navigation error in foggy conditions",
        "Fire broke out in the cargo hold during loading operations",
        "Structural damage to hull during severe weather conditions"
    ]
    
    print("\n" + "="*80)
    print("MARINE ACCIDENT SEVERITY CLASSIFICATION TEST RESULTS")
    print("="*80)
    
    for i, accident in enumerate(test_cases, 1):
        print(f"\n{i}. ACCIDENT DESCRIPTION:")
        print(f"   {accident}")
        
        try:
            result = classifier.predict_accident(accident)
            
            print(f"\n   PREDICTION: {result['severity'].upper()}")
            print(f"   CONFIDENCE: {result['confidence']:.1%}")
            print(f"   PROBABILITIES: Low: {result['probabilities'][0]:.1%}, Medium: {result['probabilities'][1]:.1%}, High: {result['probabilities'][2]:.1%}")
            print(f"   REASONING: {result['reasoning']}")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        print("-" * 80)
    
    print(f"\nTest completed! {len(test_cases)} accident scenarios analyzed.")
    print("="*80)

if __name__ == "__main__":
    test_classifier() 