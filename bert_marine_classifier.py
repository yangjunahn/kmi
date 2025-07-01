import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MarineAccidentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class BERTMarineAccidentClassifier:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    def train_model(self, X_train, y_train, epochs=10, batch_size=4, learning_rate=2e-5):
        """Train the BERT classification model"""
        print("Training BERT model...")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Create classifier
        self.classifier = BERTClassifier(self.model, num_classes=3)
        self.classifier.to(self.device)
        
        # Create dataset and dataloader
        train_dataset = MarineAccidentDataset(X_train, y_train, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=learning_rate)
        
        # Training loop
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.classifier(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
        
        print("BERT model training completed!")
        return self.classifier
    
    def predict_severity(self, text):
        """Predict severity for new accident description"""
        if self.classifier is None:
            raise ValueError("BERT model not trained yet!")
        
        # Tokenize
        encoding = self.tokenizer(
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
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
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
        reasoning = f"BERT model classified this as {severity.upper()} severity "
        reasoning += f"with {confidence:.1%} confidence.\n\n"
        
        # Add probability breakdown
        reasoning += f"Probability breakdown:\n"
        reasoning += f"• Low severity: {probabilities[0]:.1%}\n"
        reasoning += f"• Medium severity: {probabilities[1]:.1%}\n"
        reasoning += f"• High severity: {probabilities[2]:.1%}\n\n"
        
        if severity == 'high':
            if high_count > 0:
                found_words = [w for w in high_severity_words if w in text_lower]
                reasoning += f"BERT identified key indicators: '{', '.join(found_words)}' "
                reasoning += "suggesting serious consequences including:\n"
                reasoning += "• Potential loss of life or serious injuries\n"
                reasoning += "• Major property damage or environmental impact\n"
                reasoning += "• Significant operational disruptions\n"
                reasoning += "• Emergency response requirements\n"
                reasoning += "• Long-term recovery and investigation needs"
            else:
                reasoning += "BERT's contextual understanding suggests serious consequences:\n"
                reasoning += "• Safety of crew and passengers at risk\n"
                reasoning += "• Environmental protection concerns\n"
                reasoning += "• Major operational disruptions\n"
                reasoning += "• Emergency response needs\n"
                reasoning += "• Regulatory compliance and reporting requirements"
        elif severity == 'medium':
            if medium_count > 0:
                found_words = [w for w in medium_severity_words if w in text_lower]
                reasoning += f"BERT identified key indicators: '{', '.join(found_words)}' "
                reasoning += "suggesting moderate consequences including:\n"
                reasoning += "• Some operational impact but manageable\n"
                reasoning += "• Limited safety risks\n"
                reasoning += "• Minor to moderate damage\n"
                reasoning += "• Temporary operational delays\n"
                reasoning += "• Standard incident response procedures"
            else:
                reasoning += "BERT's contextual analysis suggests moderate consequences:\n"
                reasoning += "• Some operational impact but manageable\n"
                reasoning += "• Limited safety and environmental risks\n"
                reasoning += "• Moderate damage or delays\n"
                reasoning += "• Standard response procedures\n"
                reasoning += "• Routine investigation and reporting"
        else:  # low
            if low_count > 0:
                found_words = [w for w in low_severity_words if w in text_lower]
                reasoning += f"BERT identified key indicators: '{', '.join(found_words)}' "
                reasoning += "suggesting minor consequences including:\n"
                reasoning += "• Minimal operational impact\n"
                reasoning += "• Low safety risks\n"
                reasoning += "• Minor or no damage\n"
                reasoning += "• Routine operational procedures\n"
                reasoning += "• Standard maintenance or inspection procedures"
            else:
                reasoning += "BERT's contextual analysis suggests minor consequences:\n"
                reasoning += "• Minimal operational impact\n"
                reasoning += "• Low safety and environmental risks\n"
                reasoning += "• Minor damage or delays\n"
                reasoning += "• Standard operational procedures\n"
                reasoning += "• Routine documentation and reporting"
        
        # Add model confidence note
        if confidence < 0.6:
            reasoning += f"\n\nNote: BERT's confidence is relatively low ({confidence:.1%}), "
            reasoning += "suggesting the accident description may contain mixed indicators "
            reasoning += "or the situation may be borderline between severity levels."
        
        return reasoning
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the BERT model on test data"""
        print("\n=== BERT Model Evaluation ===")
        
        # Create test dataset
        test_dataset = MarineAccidentDataset(X_test, y_test, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # Predictions
        self.classifier.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.classifier(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        print(f"BERT Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['low', 'medium', 'high']))
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(all_labels, all_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['low', 'medium', 'high'],
                   yticklabels=['low', 'medium', 'high'])
        plt.title('BERT Marine Accident Severity Classification - Confusion Matrix')
        plt.ylabel('True Severity')
        plt.xlabel('Predicted Severity')
        plt.tight_layout()
        plt.savefig('bert_marine_accident_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
    
    def save_model(self, filepath_prefix):
        """Save the trained BERT model"""
        torch.save(self.classifier.state_dict(), f'{filepath_prefix}_bert_model.pth')
        self.tokenizer.save_pretrained(f'{filepath_prefix}_bert_tokenizer')
        print("BERT model saved successfully!")
    
    def load_model(self, filepath_prefix):
        """Load a trained BERT model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            self.classifier = BERTClassifier(self.model, num_classes=3)
            self.classifier.load_state_dict(torch.load(f'{filepath_prefix}_bert_model.pth', map_location=self.device))
            self.classifier.to(self.device)
            print("BERT model loaded successfully!")
        except:
            print("BERT model files not found!")

def main():
    """Main function to train and test the BERT marine accident classifier"""
    
    # Initialize classifier
    classifier = BERTMarineAccidentClassifier()
    
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
    
    # Train BERT model
    classifier.train_model(X_train, y_train, epochs=8, batch_size=2)
    
    # Evaluate model
    accuracy = classifier.evaluate_model(X_test, y_test)
    
    # Save model
    classifier.save_model('bert_marine_accident_classifier')
    
    # Test with sample accident descriptions
    print("\n" + "="*80)
    print("BERT SAMPLE PREDICTIONS")
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
        
        print(f"\n   BERT PREDICTION: {severity.upper()}")
        print(f"   CONFIDENCE: {max(probabilities):.1%}")
        print(f"\n   REASONING:")
        print(f"   {reasoning}")
        print("-" * 80)
    
    print(f"\nBERT model training and evaluation completed!")
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"BERT model saved successfully!")

if __name__ == "__main__":
    main() 