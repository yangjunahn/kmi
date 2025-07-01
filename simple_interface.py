import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class SimpleMarineInterface:
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

def main():
    """Main function"""
    interface = SimpleMarineInterface()
    
    # Train model
    interface.load_and_train('marine_accident_reports.csv')
    
    print("\n" + "="*60)
    print("MARINE ACCIDENT SEVERITY CLASSIFIER")
    print("="*60)
    print("Enter accident descriptions to get severity predictions.")
    print("Type 'quit' to exit.")
    print("="*60)
    
    while True:
        print("\n" + "-"*40)
        accident = input("Enter accident description: ").strip()
        
        if accident.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the classifier!")
            break
        
        if not accident:
            print("Please enter a valid description.")
            continue
        
        try:
            result = interface.predict_accident(accident)
            
            print(f"\nPREDICTION: {result['severity'].upper()}")
            print(f"CONFIDENCE: {result['confidence']:.1%}")
            print(f"PROBABILITIES: Low: {result['probabilities'][0]:.1%}, Medium: {result['probabilities'][1]:.1%}, High: {result['probabilities'][2]:.1%}")
            print(f"REASONING: {result['reasoning']}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 