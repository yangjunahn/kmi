#!/usr/bin/env python3
"""
Comprehensive script to train all marine accident severity classification models
and save them in organized folders for presentation and analysis.
"""

import os
import sys
import time
from datetime import datetime

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    print("✅ Directories created/verified")

def train_simple_tfidf_model():
    """Train the simple TF-IDF model"""
    print("\n" + "="*60)
    print("TRAINING SIMPLE TF-IDF MODEL")
    print("="*60)
    
    try:
        from simple_marine_classifier import SimpleMarineAccidentClassifier
        from sklearn.model_selection import train_test_split
        
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
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train model
        start_time = time.time()
        classifier.train_model(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        accuracy = classifier.evaluate_model(X_test, y_test)
        
        # Save model
        classifier.save_model('marine_accident_classifier')
        
        print(f"✅ Simple TF-IDF model trained successfully!")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Test accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training simple TF-IDF model: {e}")
        return False

def train_improved_tfidf_model():
    """Train the improved TF-IDF model with class weighting"""
    print("\n" + "="*60)
    print("TRAINING IMPROVED TF-IDF MODEL")
    print("="*60)
    
    try:
        from improved_marine_classifier import ImprovedMarineAccidentClassifier
        from sklearn.model_selection import train_test_split
        
        # Initialize classifier
        classifier = ImprovedMarineAccidentClassifier()
        
        # Load data
        df = classifier.load_data('marine_accident_reports.csv')
        
        # Prepare features and target
        X = df['Description']
        y = df['severity_numeric']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train model
        start_time = time.time()
        classifier.train_model(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        accuracy = classifier.evaluate_model(X_test, y_test)
        
        # Save model
        classifier.save_model('improved_marine_accident_classifier')
        
        print(f"✅ Improved TF-IDF model trained successfully!")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Test accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training improved TF-IDF model: {e}")
        return False

def train_bert_model():
    """Train the BERT model"""
    print("\n" + "="*60)
    print("TRAINING BERT MODEL")
    print("="*60)
    
    try:
        from bert_marine_classifier import BERTMarineAccidentClassifier
        from sklearn.model_selection import train_test_split
        
        # Initialize classifier
        classifier = BERTMarineAccidentClassifier()
        
        # Load data
        df = classifier.load_data('marine_accident_reports.csv')
        
        # Prepare features and target
        X = df['Description']
        y = df['severity_numeric']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train model (with fewer epochs for faster training)
        start_time = time.time()
        classifier.train_model(X_train, y_train, epochs=5, batch_size=2)
        training_time = time.time() - start_time
        
        # Evaluate model
        accuracy = classifier.evaluate_model(X_test, y_test)
        
        # Save model
        classifier.save_model('bert_marine_accident_classifier')
        
        print(f"✅ BERT model trained successfully!")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Test accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training BERT model: {e}")
        return False

def train_enhanced_model():
    """Train the enhanced combined model"""
    print("\n" + "="*60)
    print("TRAINING ENHANCED COMBINED MODEL")
    print("="*60)
    
    try:
        from enhanced_marine_classifier import EnhancedMarineAccidentClassifier
        from sklearn.model_selection import train_test_split
        
        # Initialize classifier
        classifier = EnhancedMarineAccidentClassifier()
        
        # Load data
        df = classifier.load_data('marine_accident_reports.csv')
        
        # Prepare features and target
        X = df['Description']
        y = df['severity_numeric']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train model
        start_time = time.time()
        classifier.train_model(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        accuracy = classifier.evaluate_model(X_test, y_test)
        
        # Save model
        classifier.save_model('enhanced_marine_accident_classifier')
        
        print(f"✅ Enhanced combined model trained successfully!")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Test accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training enhanced model: {e}")
        return False

def generate_training_summary():
    """Generate a summary of the training process"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Check which models were successfully trained
    models_status = {}
    
    # Check simple TF-IDF
    if os.path.exists('models/marine_accident_classifier_model.pkl'):
        models_status['Simple TF-IDF'] = "✅ Trained"
    else:
        models_status['Simple TF-IDF'] = "❌ Failed"
    
    # Check improved TF-IDF
    if os.path.exists('models/improved_marine_accident_classifier_model.pkl'):
        models_status['Improved TF-IDF'] = "✅ Trained"
    else:
        models_status['Improved TF-IDF'] = "❌ Failed"
    
    # Check BERT
    if os.path.exists('models/bert_marine_accident_classifier_bert_model.pth'):
        models_status['BERT'] = "✅ Trained"
    else:
        models_status['BERT'] = "❌ Failed"
    
    # Check enhanced
    if os.path.exists('models/enhanced_marine_accident_classifier_model.pkl'):
        models_status['Enhanced Combined'] = "✅ Trained"
    else:
        models_status['Enhanced Combined'] = "❌ Failed"
    
    print("Model Training Status:")
    for model, status in models_status.items():
        print(f"   {model}: {status}")
    
    # Check figures
    figures = os.listdir('figures')
    print(f"\nGenerated Figures ({len(figures)}):")
    for figure in figures:
        print(f"   📊 {figure}")
    
    # Check models
    models = os.listdir('models')
    print(f"\nSaved Models ({len(models)}):")
    for model in models:
        if model.endswith('.pkl') or model.endswith('.pth'):
            print(f"   🤖 {model}")
        else:
            print(f"   📁 {model}/")

def main():
    """Main function to train all models"""
    print("🚢 MARINE ACCIDENT SEVERITY CLASSIFICATION - MODEL TRAINING")
    print("="*70)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    create_directories()
    
    # Train models
    results = {}
    
    # Train simple TF-IDF model
    results['Simple TF-IDF'] = train_simple_tfidf_model()
    
    # Train improved TF-IDF model
    results['Improved TF-IDF'] = train_improved_tfidf_model()
    
    # Train BERT model (this might take longer)
    print("\n⚠️  BERT training may take several minutes...")
    results['BERT'] = train_bert_model()
    
    # Train enhanced model
    results['Enhanced Combined'] = train_enhanced_model()
    
    # Generate summary
    generate_training_summary()
    
    # Final status
    successful_models = sum(results.values())
    total_models = len(results)
    
    print(f"\n🎯 TRAINING COMPLETED!")
    print(f"   Successfully trained: {successful_models}/{total_models} models")
    print(f"   Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_models == total_models:
        print("   🎉 All models trained successfully!")
    else:
        print("   ⚠️  Some models failed to train. Check the error messages above.")
    
    print("\n📁 Files organized in:")
    print("   models/ - All trained model files")
    print("   figures/ - Confusion matrices and visualizations")
    
    print("\n🚀 Ready for testing! Run:")
    print("   python test_marine_classifier.py")
    print("   python demo.py")
    print("   python simple_interface.py")

if __name__ == "__main__":
    main() 