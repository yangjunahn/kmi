import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ImprovedMarineAccidentClassifier:
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
        """Train the classification model with class weights"""
        print("Training improved TF-IDF model...")
        
        # TF-IDF Vectorization with better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=1,
            max_df=0.9
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        print(f"Class weights: {weight_dict}")
        
        # Try different models with class weights
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight=weight_dict
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                random_state=42,
                class_weight=weight_dict
            )
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
        """Generate detailed reasoning for the severity prediction"""
        # Enhanced severity indicators
        high_severity_words = [
            'collision', 'fire', 'explosion', 'sinking', 'casualty', 'death', 
            'serious injury', 'major damage', 'oil spill', 'evacuation', 'multiple',
            'fatal', 'critical', 'emergency', 'disaster', 'catastrophic', 'sunk',
            'capsized', 'abandoned', 'rescue', 'hospital', 'medical'
        ]
        medium_severity_words = [
            'grounding', 'allision', 'minor damage', 'delay', 'operational issue',
            'structural damage', 'navigation error', 'equipment failure', 'contact',
            'stuck', 'aground', 'stranded', 'leak', 'flooding', 'engine failure',
            'steering failure', 'electrical failure'
        ]
        low_severity_words = [
            'minor', 'slight', 'no damage', 'operational', 'routine', 'maintenance',
            'weather', 'technical issue', 'delay', 'temporary', 'repair', 'inspection',
            'preventive', 'scheduled'
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
        
        # Add probability breakdown
        reasoning += f"Probability breakdown:\n"
        reasoning += f"• Low severity: {probabilities[0]:.1%}\n"
        reasoning += f"• Medium severity: {probabilities[1]:.1%}\n"
        reasoning += f"• High severity: {probabilities[2]:.1%}\n\n"
        
        if severity == 'high':
            if high_count > 0:
                found_words = [w for w in high_severity_words if w in text_lower]
                reasoning += f"Key indicators: The presence of terms like '{', '.join(found_words)}' "
                reasoning += "suggests serious consequences including:\n"
                reasoning += "• Potential loss of life or serious injuries\n"
                reasoning += "• Major property damage or environmental impact\n"
                reasoning += "• Significant operational disruptions\n"
                reasoning += "• Emergency response requirements\n"
                reasoning += "• Long-term recovery and investigation needs"
            else:
                reasoning += "The description suggests serious consequences that could impact:\n"
                reasoning += "• Safety of crew and passengers\n"
                reasoning += "• Environmental protection\n"
                reasoning += "• Major operational disruptions\n"
                reasoning += "• Emergency response needs\n"
                reasoning += "• Regulatory compliance and reporting"
        elif severity == 'medium':
            if medium_count > 0:
                found_words = [w for w in medium_severity_words if w in text_lower]
                reasoning += f"Key indicators: The presence of terms like '{', '.join(found_words)}' "
                reasoning += "suggests moderate consequences including:\n"
                reasoning += "• Some operational impact but manageable\n"
                reasoning += "• Limited safety risks\n"
                reasoning += "• Minor to moderate damage\n"
                reasoning += "• Temporary operational delays\n"
                reasoning += "• Standard incident response procedures"
            else:
                reasoning += "The incident appears to have moderate consequences:\n"
                reasoning += "• Some operational impact but manageable\n"
                reasoning += "• Limited safety and environmental risks\n"
                reasoning += "• Moderate damage or delays\n"
                reasoning += "• Standard response procedures\n"
                reasoning += "• Routine investigation and reporting"
        else:  # low
            if low_count > 0:
                found_words = [w for w in low_severity_words if w in text_lower]
                reasoning += f"Key indicators: The presence of terms like '{', '.join(found_words)}' "
                reasoning += "suggests minor consequences including:\n"
                reasoning += "• Minimal operational impact\n"
                reasoning += "• Low safety risks\n"
                reasoning += "• Minor or no damage\n"
                reasoning += "• Routine operational procedures\n"
                reasoning += "• Standard maintenance or inspection procedures"
            else:
                reasoning += "The incident appears to have minor consequences:\n"
                reasoning += "• Minimal operational impact\n"
                reasoning += "• Low safety and environmental risks\n"
                reasoning += "• Minor damage or delays\n"
                reasoning += "• Standard operational procedures\n"
                reasoning += "• Routine documentation and reporting"
        
        # Add model confidence note
        if confidence < 0.6:
            reasoning += f"\n\nNote: The model's confidence is relatively low ({confidence:.1%}), "
            reasoning += "suggesting the accident description may contain mixed indicators "
            reasoning += "or the situation may be borderline between severity levels."
        
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
        plt.title('Improved Marine Accident Severity Classification - Confusion Matrix')
        plt.ylabel('True Severity')
        plt.xlabel('Predicted Severity')
        plt.tight_layout()
        plt.savefig('figures/improved_marine_accident_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
    
    def save_model(self, filepath_prefix):
        """Save the trained model"""
        import joblib
        
        joblib.dump(self.vectorizer, f'models/{filepath_prefix}_vectorizer.pkl')
        joblib.dump(self.model, f'models/{filepath_prefix}_model.pkl')
        print("Model saved successfully!")
    
    def load_model(self, filepath_prefix):
        """Load a trained model"""
        import joblib
        
        try:
            self.vectorizer = joblib.load(f'models/{filepath_prefix}_vectorizer.pkl')
            self.model = joblib.load(f'models/{filepath_prefix}_model.pkl')
            print("Model loaded successfully!")
        except:
            print("Model files not found!")

def main():
    """Main function to train and test the improved marine accident classifier"""
    
    # Initialize classifier
    classifier = ImprovedMarineAccidentClassifier()
    
    # Load data
    df = classifier.load_data('marine_accident_reports.csv')
    
    # Prepare features and target
    X = df['Description']
    y = df['severity_numeric']
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    classifier.train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = classifier.evaluate_model(X_test, y_test)
    
    # Save model
    classifier.save_model('improved_marine_accident_classifier')
    
    # Test with sample accident descriptions
    print("\n" + "="*80)
    print("IMPROVED SAMPLE PREDICTIONS")
    print("="*80)
    
    sample_accidents = [
        "Collision between two cargo ships in the harbor resulting in major hull damage and oil spill",
        "Minor grounding incident with no damage to vessel or environment",
        "Fire outbreak in engine room causing evacuation of crew members",
        "Allision with dock structure causing minor damage to bow",
        "Explosion in cargo hold leading to multiple casualties",
        "Navigation error causing vessel to run aground in shallow water",
        "Routine maintenance issue causing temporary operational delay",
        "Weather-related incident causing minor structural damage",
        "Major collision with bridge structure causing vessel to sink",
        "Engine room fire resulting in complete power loss and emergency response"
    ]
    
    for i, accident in enumerate(sample_accidents, 1):
        print(f"\n{i}. ACCIDENT DESCRIPTION:")
        print(f"   {accident}")
        
        # Predict severity
        severity, probabilities = classifier.predict_severity(accident)
        reasoning = classifier.generate_reasoning(accident, severity, probabilities)
        
        print(f"\n   PREDICTION: {severity.upper()}")
        print(f"   CONFIDENCE: {max(probabilities):.1%}")
        print(f"\n   REASONING:")
        print(f"   {reasoning}")
        print("-" * 80)
    
    print(f"\nImproved model training and evaluation completed!")
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"Model saved as 'improved_marine_accident_classifier_model.pkl' and 'improved_marine_accident_classifier_vectorizer.pkl'")

if __name__ == "__main__":
    main() 