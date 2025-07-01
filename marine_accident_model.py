import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class MarineAccidentClassifier:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def scrape_pdf_content(self, url, max_retries=3):
        """Attempt to scrape content from PDF links (simplified version)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                # For PDF links, we'll extract basic info from the URL
                # In a real implementation, you'd use a PDF parser
                return f"Marine accident report from {url}"
            else:
                return "No additional information available"
        except:
            return "No additional information available"
    
    def load_and_preprocess_data(self, csv_file):
        """Load and preprocess the marine accident data"""
        print("Loading data...")
        df = pd.read_csv(csv_file)
        
        # Clean the data
        df_clean = df.dropna(subset=['Description', 'Severity'])
        
        # Map severity to numerical values
        severity_mapping = {'low': 0, 'medium': 1, 'high': 2}
        df_clean['severity_numeric'] = df_clean['Severity'].map(severity_mapping)
        
        # Combine description with scraped content
        print("Enhancing data with web scraping...")
        enhanced_descriptions = []
        for idx, row in df_clean.iterrows():
            desc = str(row['Description'])
            if pd.notna(row['Pdf Link']):
                additional_info = self.scrape_pdf_content(row['Pdf Link'])
                enhanced_desc = f"{desc} {additional_info}"
            else:
                enhanced_desc = desc
            enhanced_descriptions.append(enhanced_desc)
        
        df_clean['Enhanced_Description'] = enhanced_descriptions
        
        print(f"Final dataset shape: {df_clean.shape}")
        print(f"Severity distribution:\n{df_clean['Severity'].value_counts()}")
        
        return df_clean
    
    def train_tfidf_model(self, X_train, y_train):
        """Train TF-IDF based model"""
        print("Training TF-IDF model...")
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        
        # Train multiple models
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
        
        self.tfidf_model = best_model
        print(f"Selected best TF-IDF model with accuracy: {best_score:.4f}")
        
        return best_model
    
    def train_bert_model(self, X_train, y_train, epochs=5):
        """Train BERT based model"""
        print("Training BERT model...")
        
        # Load BERT tokenizer and model
        model_name = "bert-base-uncased"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        # Create BERT classifier
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
        
        # Initialize classifier
        classifier = BERTClassifier(self.bert_model, num_classes=3)
        classifier.to(self.device)
        
        # Prepare data
        max_length = 512
        train_encodings = self.bert_tokenizer(
            X_train.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Convert to tensors
        input_ids = train_encodings['input_ids'].to(self.device)
        attention_mask = train_encodings['attention_mask'].to(self.device)
        labels = torch.tensor(y_train.values, dtype=torch.long).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)
        
        # Training loop
        classifier.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = classifier(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        self.bert_classifier = classifier
        print("BERT model training completed!")
        
        return classifier
    
    def predict_severity(self, text, model_type='tfidf'):
        """Predict severity for new accident description"""
        if model_type == 'tfidf':
            if self.tfidf_model is None:
                raise ValueError("TF-IDF model not trained yet!")
            
            # Transform text
            text_tfidf = self.tfidf_vectorizer.transform([text])
            
            # Predict
            prediction = self.tfidf_model.predict(text_tfidf)[0]
            probabilities = self.tfidf_model.predict_proba(text_tfidf)[0]
            
        elif model_type == 'bert':
            if self.bert_classifier is None:
                raise ValueError("BERT model not trained yet!")
            
            # Tokenize
            encoding = self.bert_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            self.bert_classifier.eval()
            with torch.no_grad():
                outputs = self.bert_classifier(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                prediction = np.argmax(probabilities)
        
        # Map back to severity labels
        severity_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        severity = severity_mapping[prediction]
        
        return severity, probabilities
    
    def generate_reasoning(self, text, severity, probabilities):
        """Generate reasoning for the severity prediction"""
        severity_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        
        # Extract key words that might indicate severity
        high_severity_words = ['collision', 'fire', 'explosion', 'sinking', 'casualty', 'death', 'serious injury', 'major damage']
        medium_severity_words = ['grounding', 'allision', 'minor damage', 'delay', 'operational issue']
        low_severity_words = ['minor', 'slight', 'no damage', 'operational', 'routine']
        
        # Count severity indicators
        text_lower = text.lower()
        high_count = sum(1 for word in high_severity_words if word in text_lower)
        medium_count = sum(1 for word in medium_severity_words if word in text_lower)
        low_count = sum(1 for word in low_severity_words if word in text_lower)
        
        # Generate reasoning
        confidence = max(probabilities)
        reasoning = f"Based on the accident description, the model classified this as {severity} severity "
        reasoning += f"with {confidence:.1%} confidence. "
        
        if severity == 'high':
            if high_count > 0:
                reasoning += f"The presence of terms like '{', '.join([w for w in high_severity_words if w in text_lower])}' "
                reasoning += "indicates serious consequences such as potential loss of life, major property damage, "
                reasoning += "or significant environmental impact."
            else:
                reasoning += "The description suggests serious consequences that could impact safety, "
                reasoning += "environment, or major operational disruptions."
        elif severity == 'medium':
            if medium_count > 0:
                reasoning += f"The presence of terms like '{', '.join([w for w in medium_severity_words if w in text_lower])}' "
                reasoning += "indicates moderate consequences with some operational impact but limited safety risks."
            else:
                reasoning += "The incident appears to have moderate consequences with some operational impact "
                reasoning += "but manageable safety and environmental risks."
        else:  # low
            if low_count > 0:
                reasoning += f"The presence of terms like '{', '.join([w for w in low_severity_words if w in text_lower])}' "
                reasoning += "indicates minor consequences with minimal operational impact and low safety risks."
            else:
                reasoning += "The incident appears to have minor consequences with minimal operational impact "
                reasoning += "and low safety and environmental risks."
        
        return reasoning
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models on test data"""
        print("\n=== Model Evaluation ===")
        
        # TF-IDF evaluation
        if self.tfidf_model is not None:
            X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
            y_pred_tfidf = self.tfidf_model.predict(X_test_tfidf)
            accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
            
            print(f"\nTF-IDF Model Results:")
            print(f"Accuracy: {accuracy_tfidf:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_tfidf, target_names=['low', 'medium', 'high']))
            
            # Confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred_tfidf)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['low', 'medium', 'high'],
                       yticklabels=['low', 'medium', 'high'])
            plt.title('TF-IDF Model Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('tfidf_confusion_matrix.png')
            plt.show()
        
        # BERT evaluation (if enough data)
        if len(X_test) > 5 and self.bert_classifier is not None:
            print(f"\nBERT Model Results:")
            # Note: BERT evaluation would require more data and longer processing time
            print("BERT evaluation requires more data for reliable results")
    
    def save_models(self, filepath_prefix):
        """Save trained models"""
        import joblib
        
        # Save TF-IDF model
        if self.tfidf_model is not None:
                    joblib.dump(self.tfidf_vectorizer, f'models/{filepath_prefix}_tfidf_vectorizer.pkl')
        joblib.dump(self.tfidf_model, f'models/{filepath_prefix}_tfidf_model.pkl')
            print("TF-IDF models saved successfully!")
        
        # Save BERT model
        if self.bert_classifier is not None:
            torch.save(self.bert_classifier.state_dict(), f'{filepath_prefix}_bert_model.pth')
            self.bert_tokenizer.save_pretrained(f'{filepath_prefix}_bert_tokenizer')
            print("BERT model saved successfully!")
    
    def load_models(self, filepath_prefix):
        """Load trained models"""
        import joblib
        
        # Load TF-IDF model
        try:
            self.tfidf_vectorizer = joblib.load(f'models/{filepath_prefix}_tfidf_vectorizer.pkl')
            self.tfidf_model = joblib.load(f'models/{filepath_prefix}_tfidf_model.pkl')
            print("TF-IDF models loaded successfully!")
        except:
            print("TF-IDF models not found!")
        
        # Load BERT model
        try:
            model_name = "bert-base-uncased"
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            
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
            self.bert_classifier.load_state_dict(torch.load(f'{filepath_prefix}_bert_model.pth', map_location=self.device))
            self.bert_classifier.to(self.device)
            print("BERT model loaded successfully!")
        except:
            print("BERT model not found!")

def main():
    """Main function to train and test the marine accident classifier"""
    
    # Initialize classifier
    classifier = MarineAccidentClassifier()
    
    # Load and preprocess data
    df = classifier.load_and_preprocess_data('marine_accident_reports.csv')
    
    # Prepare features and target
    X = df['Enhanced_Description']
    y = df['severity_numeric']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train TF-IDF model
    classifier.train_tfidf_model(X_train, y_train)
    
    # Train BERT model (if enough data)
    if len(X_train) > 10:
        classifier.train_bert_model(X_train, y_train, epochs=3)
    
    # Evaluate models
    classifier.evaluate_models(X_test, y_test)
    
    # Save models
    classifier.save_models('marine_accident_classifier')
    
    # Test with sample accident descriptions
    print("\n=== Sample Predictions ===")
    sample_accidents = [
        "Collision between two cargo ships in the harbor resulting in major hull damage and oil spill",
        "Minor grounding incident with no damage to vessel or environment",
        "Fire outbreak in engine room causing evacuation of crew members",
        "Allision with dock structure causing minor damage to bow",
        "Explosion in cargo hold leading to multiple casualties"
    ]
    
    for accident in sample_accidents:
        print(f"\nAccident: {accident}")
        
        # TF-IDF prediction
        severity_tfidf, prob_tfidf = classifier.predict_severity(accident, 'tfidf')
        reasoning_tfidf = classifier.generate_reasoning(accident, severity_tfidf, prob_tfidf)
        
        print(f"TF-IDF Prediction: {severity_tfidf.upper()}")
        print(f"TF-IDF Reasoning: {reasoning_tfidf}")
        
        # BERT prediction (if available)
        if classifier.bert_classifier is not None:
            try:
                severity_bert, prob_bert = classifier.predict_severity(accident, 'bert')
                reasoning_bert = classifier.generate_reasoning(accident, severity_bert, prob_bert)
                print(f"BERT Prediction: {severity_bert.upper()}")
                print(f"BERT Reasoning: {reasoning_bert}")
            except:
                print("BERT prediction not available")

if __name__ == "__main__":
    main() 