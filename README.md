# Marine Accident Severity Classification System

## Project Overview

This project implements an AI model for classifying marine accident severity based on accident descriptions. The system can predict whether a marine accident is of "low", "medium", or "high" severity and provides detailed reasoning for its predictions.

## Features

- **TF-IDF based classification**: Uses TF-IDF vectorization with Random Forest classifier
- **BERT-based classification**: Advanced transformer model for better text understanding
- **Detailed reasoning**: Provides explanations for severity predictions
- **Multiple model comparison**: Compare TF-IDF and BERT approaches
- **Interactive interface**: User-friendly command-line interface for testing
- **Comprehensive evaluation**: Detailed analysis of model performance

## Requirements

- Python 3.8+
- Required packages (see requirements.txt):
  - pandas
  - scikit-learn
  - numpy
  - matplotlib
  - seaborn
  - transformers
  - torch
  - requests
  - beautifulsoup4

## Installation

1. Create a virtual environment:
```bash
python3 -m venv marine_accident_env
source marine_accident_env/bin/activate  # On Windows: marine_accident_env\Scripts\activate
```

2. Install required packages:
```bash
pip install pandas scikit-learn matplotlib seaborn numpy requests beautifulsoup4 transformers torch
```

3. Ensure you have the `marine_accident_reports.csv` file in the project directory.

## Usage

### 1. Simple TF-IDF Classifier
```bash
python3 simple_marine_classifier.py
```
This runs the basic TF-IDF based classifier with model evaluation and sample predictions.

### 2. Improved TF-IDF Classifier
```bash
python3 improved_marine_classifier.py
```
This runs an enhanced version with better class balancing and feature engineering.

### 3. BERT-based Classifier
```bash
python3 bert_marine_classifier.py
```
This runs the BERT transformer model for more sophisticated text understanding.

### 4. Interactive Interface
```bash
python3 simple_interface.py
```
This provides an interactive command-line interface where you can input accident descriptions and get predictions.

### 5. Test Suite
```bash
python3 test_marine_classifier.py
```
This runs a comprehensive test with predefined accident scenarios.

### 6. Enhanced Test Suite
```bash
python3 enhanced_marine_classifier.py
```
This runs an enhanced test with expected severity levels and accuracy evaluation.

## Data Structure

The system expects a CSV file (`marine_accident_reports.csv`) with the following columns:
- `Date`: Date of the accident
- `Pdf Link`: Link to detailed report (optional)
- `Description`: Text description of the accident
- `Severity`: Expert-assigned severity (low/medium/high)
- `Reason`: Expert reasoning for severity assignment (optional)

## Model Performance

### Current Limitations
- **Limited training data**: Only 42 records with severity labels
- **Class imbalance**: Most accidents are classified as "low" severity
- **Data quality**: Many records lack severity information

### Performance Metrics
- **TF-IDF Model**: ~60-65% test accuracy
- **BERT Model**: ~60-65% test accuracy
- **Enhanced Model**: ~27% accuracy on test scenarios

### Recommendations for Improvement
1. **More training data**: Collect more marine accident reports with severity labels
2. **Data augmentation**: Use techniques to balance the dataset
3. **Feature engineering**: Extract more domain-specific features
4. **Ensemble methods**: Combine multiple models for better predictions
5. **Domain expertise**: Incorporate expert knowledge into the model

## Example Usage

### Python API
```python
from enhanced_marine_classifier import EnhancedMarineAccidentClassifier

# Initialize and train
classifier = EnhancedMarineAccidentClassifier()
classifier.load_and_train('marine_accident_reports.csv')

# Predict severity
accident = "Collision between two cargo ships in the harbor resulting in major hull damage and oil spill"
result = classifier.predict_accident(accident)

print(f"Severity: {result['severity']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Reasoning: {result['reasoning']}")
```

### Command Line
```bash
python3 simple_interface.py
# Then enter accident descriptions interactively
```

## Severity Classification Criteria

### High Severity Indicators
- Collision, fire, explosion
- Sinking, casualties, death
- Major damage, oil spill
- Evacuation, emergency response
- Missing persons, fatalities

### Medium Severity Indicators
- Grounding, allision
- Minor damage, structural damage
- Navigation errors, equipment failure
- Operational delays, contact incidents

### Low Severity Indicators
- Minor, slight, no damage
- Routine maintenance, operational issues
- Weather-related incidents
- Technical issues, temporary delays

## Project Structure

```
marine_accident_classification/
├── marine_accident_reports.csv      # Training data
├── simple_marine_classifier.py      # Basic TF-IDF classifier
├── improved_marine_classifier.py    # Enhanced TF-IDF classifier
├── bert_marine_classifier.py        # BERT-based classifier
├── simple_interface.py              # Interactive interface
├── test_marine_classifier.py        # Test suite
├── enhanced_marine_classifier.py    # Enhanced test suite
├── marine_accident_model.py         # Comprehensive model (TF-IDF + BERT)
└── README.md                        # This file
```

## Model Files Generated

After training, the following files are created:
- `*_vectorizer.pkl`: TF-IDF vectorizer
- `*_model.pkl`: Trained classifier model
- `*_bert_model.pth`: BERT model weights
- `*_bert_tokenizer/`: BERT tokenizer files
- `*_confusion_matrix.png`: Confusion matrix visualization

## Future Improvements

1. **Data Collection**: Gather more marine accident reports with expert severity assessments
2. **Feature Engineering**: Extract more domain-specific features (vessel type, location, weather conditions)
3. **Model Architecture**: Try different transformer models (RoBERTa, DistilBERT)
4. **Ensemble Methods**: Combine multiple models for better predictions
5. **Real-time Processing**: Implement real-time accident description processing
6. **Web Interface**: Create a web-based interface for easier access
7. **API Development**: Develop REST API for integration with other systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with data privacy and usage regulations.

## Contact

For questions or contributions, please open an issue in the repository.

---

**Note**: This system is designed for educational and research purposes. For real-world marine safety applications, always consult with maritime safety experts and follow established protocols and regulations. 