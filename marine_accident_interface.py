import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MarineAccidentInterface:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.bert_classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.severity_mapping = {'low': 0, 'medium': 1, 'high': 2}
        self.reverse_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        
    def load_and_train_models(self, csv_file):
        """Load data and train both TF-IDF and BERT models"""
        print("="*60)
        print("MARINE ACCIDENT SEVERITY CLASSIFICATION SYSTEM")
        print("="*60)
        
        # Load data
        print("\n1. Loading marine accident data...")
        df = pd.read_csv(csv_file)
        df_clean = df.dropna(subset=['Description', 'Severity'])
        df_clean['severity_numeric'] = df_clean['Severity'].map(self.severity_mapping)
        
        print(f"   Original dataset: {df.shape[0]} records")
        print(f"   Clean dataset: {df_clean.shape[0]} records")
        print(f"   Severity distribution: {df_clean['Severity'].value_counts().to_dict()}")
        
        # Prepare data
        X = df_clean['Description']
        y = df_clean['severity_numeric']
        
        # Train TF-IDF model
        print("\n2. Training TF-IDF model...")
        self.train_tfidf_model(X, y)
        
        # Train BERT model
        print("\n3. Training BERT model...")
        self.train_bert_model(X, y)
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED!")
        print("="*60)
    
    def train_tfidf_model(self, X, y):
        """Train TF-IDF based model"""
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9
        )
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        weight_dict = dict(zip(np.unique(y), class_weights))
        
        # Train model
        self.tfidf_model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42,
            class_weight=weight_dict
        )
        self.tfidf_model.fit(X_tfidf, y)
        
        # Evaluate
        y_pred = self.tfidf_model.predict(X_tfidf)
        accuracy = accuracy_score(y, y_pred)
        print(f"   TF-IDF training accuracy: {accuracy:.4f}")
    
    def train_bert_model(self, X, y, epochs=5):
        """Train BERT based model"""
        # Load BERT
        model_name = "bert-base-uncased"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        # Create classifier
        class BERTClassifier(nn.Module):
            def __init__(self, bert_model, num_classes):
                super(BERTClassifier, self).__init__()
                self.bert = bert_model
                self.dropout = nn.Dropout(0.3)
                self.classifier = nn.Linear(768, num_classes)
                
            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                pooled_output = self.dropout(pooled_output)
                return self.classifier(pooled_output)
        
        self.bert_classifier = BERTClassifier(self.bert_model, num_classes=3)
        self.bert_classifier.to(self.device)
        
        # Prepare data
        max_length = 512
        train_encodings = self.bert_tokenizer(
            X.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Convert to tensors
        input_ids = train_encodings['input_ids'].to(self.device)
        attention_mask = train_encodings['attention_mask'].to(self.device)
        labels = torch.tensor(y.values, dtype=torch.long).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.bert_classifier.parameters(), lr=2e-5)
        
        # Training loop
        self.bert_classifier.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.bert_classifier(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                print(f"   BERT Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        print(f"   BERT training completed!")
    
    def predict_accident_severity(self, accident_description):
        """Predict severity using both models"""
        print(f"\n{'='*80}")
        print(f"ACCIDENT ANALYSIS")
        print(f"{'='*80}")
        print(f"Description: {accident_description}")
        print(f"{'='*80}")
        
        # TF-IDF prediction
        print("\n1. TF-IDF MODEL ANALYSIS:")
        severity_tfidf, prob_tfidf = self.predict_tfidf(accident_description)
        reasoning_tfidf = self.generate_reasoning(accident_description, severity_tfidf, prob_tfidf, "TF-IDF")
        
        print(f"   Prediction: {severity_tfidf.upper()}")
        print(f"   Confidence: {max(prob_tfidf):.1%}")
        print(f"   Probabilities: Low: {prob_tfidf[0]:.1%}, Medium: {prob_tfidf[1]:.1%}, High: {prob_tfidf[2]:.1%}")
        print(f"   Reasoning: {reasoning_tfidf}")
        
        # BERT prediction
        print("\n2. BERT MODEL ANALYSIS:")
        severity_bert, prob_bert = self.predict_bert(accident_description)
        reasoning_bert = self.generate_reasoning(accident_description, severity_bert, prob_bert, "BERT")
        
        print(f"   Prediction: {severity_bert.upper()}")
        print(f"   Confidence: {max(prob_bert):.1%}")
        print(f"   Probabilities: Low: {prob_bert[0]:.1%}, Medium: {prob_bert[1]:.1%}, High: {prob_bert[2]:.1%}")
        print(f"   Reasoning: {reasoning_bert}")
        
        # Model comparison
        print(f"\n3. MODEL COMPARISON:")
        if severity_tfidf == severity_bert:
            print(f"   ✓ Both models agree: {severity_tfidf.upper()} severity")
        else:
            print(f"   ⚠ Models disagree: TF-IDF predicts {severity_tfidf.upper()}, BERT predicts {severity_bert.upper()}")
        
        confidence_diff = abs(max(prob_tfidf) - max(prob_bert))
        if confidence_diff > 0.2:
            print(f"   ⚠ Significant confidence difference: {confidence_diff:.1%}")
        
        # Final recommendation
        print(f"\n4. FINAL RECOMMENDATION:")
        if severity_tfidf == severity_bert:
            final_severity = severity_tfidf
            confidence = (max(prob_tfidf) + max(prob_bert)) / 2
            print(f"   Recommended Severity: {final_severity.upper()}")
            print(f"   Average Confidence: {confidence:.1%}")
        else:
            # Use the model with higher confidence
            if max(prob_tfidf) > max(prob_bert):
                final_severity = severity_tfidf
                confidence = max(prob_tfidf)
                model_used = "TF-IDF"
            else:
                final_severity = severity_bert
                confidence = max(prob_bert)
                model_used = "BERT"
            print(f"   Recommended Severity: {final_severity.upper()} (based on {model_used} model)")
            print(f"   Confidence: {confidence:.1%}")
        
        return final_severity, confidence
    
    def predict_tfidf(self, text):
        """Predict using TF-IDF model"""
        text_tfidf = self.tfidf_vectorizer.transform([text])
        prediction = self.tfidf_model.predict(text_tfidf)[0]
        probabilities = self.tfidf_model.predict_proba(text_tfidf)[0]
        severity = self.reverse_mapping[prediction]
        return severity, probabilities
    
    def predict_bert(self, text):
        """Predict using BERT model"""
        encoding = self.bert_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        self.bert_classifier.eval()
        with torch.no_grad():
            outputs = self.bert_classifier(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        severity = self.reverse_mapping[prediction]
        return severity, probabilities
    
    def generate_reasoning(self, text, severity, probabilities, model_name):
        """Generate reasoning for the severity prediction"""
        # Severity indicators
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
        
        # Count indicators
        text_lower = text.lower()
        high_count = sum(1 for word in high_severity_words if word in text_lower)
        medium_count = sum(1 for word in medium_severity_words if word in text_lower)
        low_count = sum(1 for word in low_severity_words if word in text_lower)
        
        confidence = max(probabilities)
        
        if severity == 'high':
            if high_count > 0:
                found_words = [w for w in high_severity_words if w in text_lower]
                return f"{model_name} identified high-severity indicators: '{', '.join(found_words)}'. Suggests serious consequences with potential loss of life, major damage, or environmental impact."
            else:
                return f"{model_name}'s contextual analysis suggests serious consequences requiring emergency response and major operational disruptions."
        elif severity == 'medium':
            if medium_count > 0:
                found_words = [w for w in medium_severity_words if w in text_lower]
                return f"{model_name} identified medium-severity indicators: '{', '.join(found_words)}'. Suggests moderate consequences with manageable operational impact."
            else:
                return f"{model_name}'s analysis suggests moderate consequences with some operational impact but limited safety risks."
        else:  # low
            if low_count > 0:
                found_words = [w for w in low_severity_words if w in text_lower]
                return f"{model_name} identified low-severity indicators: '{', '.join(found_words)}'. Suggests minor consequences with minimal operational impact."
            else:
                return f"{model_name}'s analysis suggests minor consequences with routine operational procedures."
    
    def interactive_mode(self):
        """Run interactive mode for user input"""
        print("\n" + "="*80)
        print("INTERACTIVE MARINE ACCIDENT SEVERITY CLASSIFIER")
        print("="*80)
        print("Enter accident descriptions to get severity predictions.")
        print("Type 'quit' to exit.")
        print("="*80)
        
        while True:
            print("\n" + "-"*60)
            accident = input("Enter accident description: ").strip()
            
            if accident.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using the Marine Accident Severity Classifier!")
                break
            
            if not accident:
                print("Please enter a valid accident description.")
                continue
            
            try:
                severity, confidence = self.predict_accident_severity(accident)
                print(f"\n{'='*60}")
                print(f"FINAL RESULT: {severity.upper()} SEVERITY ({confidence:.1%} confidence)")
                print(f"{'='*60}")
            except Exception as e:
                print(f"Error processing accident description: {e}")
                print("Please try again with a different description.")

def main():
    """Main function to run the marine accident interface"""
    
    # Initialize interface
    interface = MarineAccidentInterface()
    
    # Load and train models
    interface.load_and_train_models('marine_accident_reports.csv')
    
    # Run interactive mode
    interface.interactive_mode()

if __name__ == "__main__":
    main() 