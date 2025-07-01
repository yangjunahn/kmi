import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleMarineAccidentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.severity_mapping = {'low': 0, 'medium': 1, 'high': 2}
        self.reverse_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        
    def load_data(self, csv_file):
        """Load and clean the marine accident data"""
        print("Loading marine accident data...")
        df = pd.read_csv(csv_file)
        
        # Clean the data - keep only rows with both Description and Severity
        df_clean = df.dropna(subset=['Description', 'Severity'])
        
        # Convert severity to numerical
        df_clean['severity_numeric'] = df_clean['Severity'].map(self.severity_mapping)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Clean dataset shape: {df_clean.shape}")
        print(f"Severity distribution:\n{df_clean['Severity'].value_counts()}")
        
        return df_clean
    
    def train_model(self, X_train, y_train):
        """Train the classification model"""
        print("Training TF-IDF model...")
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Try different models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            score = model.score(X_train_tfidf, y_train)
            print(f"{name} training accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.model = best_model
        print(f"Selected best model with accuracy: {best_score:.4f}")
        
        return best_model
    
    def predict_severity(self, text):
        """Predict severity for new accident description"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Transform text
        text_tfidf = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Map back to severity labels
        severity = self.reverse_mapping[prediction]
        
        return severity, probabilities
    
    def generate_reasoning(self, text, severity, probabilities):
        """Generate reasoning for the severity prediction"""
        # Define severity indicators
        high_severity_words = [
            'collision', 'fire', 'explosion', 'sinking', 'casualty', 'death', 
            'serious injury', 'major damage', 'oil spill', 'evacuation', 'multiple'
        ]
        medium_severity_words = [
            'grounding', 'allision', 'minor damage', 'delay', 'operational issue',
            'structural damage', 'navigation error', 'equipment failure'
        ]
        low_severity_words = [
            'minor', 'slight', 'no damage', 'operational', 'routine', 'maintenance',
            'weather', 'technical issue'
        ]
        
        # Count severity indicators
        text_lower = text.lower()
        high_count = sum(1 for word in high_severity_words if word in text_lower)
        medium_count = sum(1 for word in medium_severity_words if word in text_lower)
        low_count = sum(1 for word in low_severity_words if word in text_lower)
        
        # Generate reasoning
        confidence = max(probabilities)
        reasoning = f"Based on the accident description, the model classified this as {severity.upper()} severity "
        reasoning += f"with {confidence:.1%} confidence.\n\n"
        
        if severity == 'high':
            if high_count > 0:
                found_words = [w for w in high_severity_words if w in text_lower]
                reasoning += f"Key indicators: The presence of terms like '{', '.join(found_words)}' "
                reasoning += "suggests serious consequences including:\n"
                reasoning += "• Potential loss of life or serious injuries\n"
                reasoning += "• Major property damage or environmental impact\n"
                reasoning += "• Significant operational disruptions\n"
                reasoning += "• Emergency response requirements"
            else:
                reasoning += "The description suggests serious consequences that could impact:\n"
                reasoning += "• Safety of crew and passengers\n"
                reasoning += "• Environmental protection\n"
                reasoning += "• Major operational disruptions\n"
                reasoning += "• Emergency response needs"
        elif severity == 'medium':
            if medium_count > 0:
                found_words = [w for w in medium_severity_words if w in text_lower]
                reasoning += f"Key indicators: The presence of terms like '{', '.join(found_words)}' "
                reasoning += "suggests moderate consequences including:\n"
                reasoning += "• Some operational impact but manageable\n"
                reasoning += "• Limited safety risks\n"
                reasoning += "• Minor to moderate damage\n"
                reasoning += "• Temporary operational delays"
            else:
                reasoning += "The incident appears to have moderate consequences:\n"
                reasoning += "• Some operational impact but manageable\n"
                reasoning += "• Limited safety and environmental risks\n"
                reasoning += "• Moderate damage or delays\n"
                reasoning += "• Standard response procedures"
        else:  # low
            if low_count > 0:
                found_words = [w for w in low_severity_words if w in text_lower]
                reasoning += f"Key indicators: The presence of terms like '{', '.join(found_words)}' "
                reasoning += "suggests minor consequences including:\n"
                reasoning += "• Minimal operational impact\n"
                reasoning += "• Low safety risks\n"
                reasoning += "• Minor or no damage\n"
                reasoning += "• Routine operational procedures"
            else:
                reasoning += "The incident appears to have minor consequences:\n"
                reasoning += "• Minimal operational impact\n"
                reasoning += "• Low safety and environmental risks\n"
                reasoning += "• Minor damage or delays\n"
                reasoning += "• Standard operational procedures"
        
        return reasoning
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on test data"""
        print("\n=== Model Evaluation ===")
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['low', 'medium', 'high']))
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['low', 'medium', 'high'],
                   yticklabels=['low', 'medium', 'high'])
        plt.title('Marine Accident Severity Classification - Confusion Matrix')
        plt.ylabel('True Severity')
        plt.xlabel('Predicted Severity')
        plt.tight_layout()
        plt.savefig('marine_accident_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
    
    def save_model(self, filepath_prefix):
        """Save the trained model"""
        import joblib
        
        joblib.dump(self.vectorizer, f'{filepath_prefix}_vectorizer.pkl')
        joblib.dump(self.model, f'{filepath_prefix}_model.pkl')
        print("Model saved successfully!")
    
    def load_model(self, filepath_prefix):
        """Load a trained model"""
        import joblib
        
        try:
            self.vectorizer = joblib.load(f'{filepath_prefix}_vectorizer.pkl')
            self.model = joblib.load(f'{filepath_prefix}_model.pkl')
            print("Model loaded successfully!")
        except:
            print("Model files not found!")

def main():
    """Main function to train and test the marine accident classifier"""
    
    # Initialize classifier
    classifier = SimpleMarineAccidentClassifier()
    
    # Load data
    df = classifier.load_data('marine_accident_reports.csv')
    
    # Prepare features and target
    X = df['Description']
    y = df['severity_numeric']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    classifier.train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = classifier.evaluate_model(X_test, y_test)
    
    # Save model
    classifier.save_model('marine_accident_classifier')
    
    # Test with sample accident descriptions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    sample_accidents = [
        "Collision between two cargo ships in the harbor resulting in major hull damage and oil spill",
        "Minor grounding incident with no damage to vessel or environment",
        "Fire outbreak in engine room causing evacuation of crew members",
        "Allision with dock structure causing minor damage to bow",
        "Explosion in cargo hold leading to multiple casualties",
        "Navigation error causing vessel to run aground in shallow water",
        "Routine maintenance issue causing temporary operational delay",
        "Weather-related incident causing minor structural damage"
    ]
    
    for i, accident in enumerate(sample_accidents, 1):
        print(f"\n{i}. ACCIDENT DESCRIPTION:")
        print(f"   {accident}")
        
        # Predict severity
        severity, probabilities = classifier.predict_severity(accident)
        reasoning = classifier.generate_reasoning(accident, severity, probabilities)
        
        print(f"\n   PREDICTION: {severity.upper()}")
        print(f"   CONFIDENCE: {max(probabilities):.1%}")
        print(f"   PROBABILITIES: Low: {probabilities[0]:.1%}, Medium: {probabilities[1]:.1%}, High: {probabilities[2]:.1%}")
        print(f"\n   REASONING:")
        print(f"   {reasoning}")
        print("-" * 80)
    
    print(f"\nModel training and evaluation completed!")
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"Model saved as 'marine_accident_classifier_model.pkl' and 'marine_accident_classifier_vectorizer.pkl'")

if __name__ == "__main__":
    main() 