# Marine Accident Severity Classifier - GitHub Pages Interface

This repository includes a web interface for the Marine Accident Severity Classifier that can be deployed on GitHub Pages.

## 🚀 Quick Start - Deploy to GitHub Pages

### Option 1: Automatic Deployment (Recommended)

1. **Push the files to your GitHub repository**
   ```bash
   git add index.html README_GITHUB_PAGES.md
   git commit -m "Add web interface for GitHub Pages"
   git push origin main
   ```

2. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Click on "Settings" tab
   - Scroll down to "Pages" section
   - Under "Source", select "Deploy from a branch"
   - Choose "main" branch and "/ (root)" folder
   - Click "Save"

3. **Access your interface**
   - Your interface will be available at: `https://yourusername.github.io/your-repo-name/`
   - It may take a few minutes to deploy

### Option 2: Manual Setup

1. **Create a new branch for GitHub Pages**
   ```bash
   git checkout -b gh-pages
   git add index.html
   git commit -m "Add web interface"
   git push origin gh-pages
   ```

2. **Enable GitHub Pages for gh-pages branch**
   - Go to repository Settings → Pages
   - Select "Deploy from a branch"
   - Choose "gh-pages" branch
   - Save

## 🔧 Customization Options

### Option A: Simple Static Interface (Current)
- ✅ **Pros**: Fast, free, no server needed
- ❌ **Cons**: Uses simulated results (not real ML models)
- 📁 **Files**: `index.html` only

### Option B: Connect to Real ML Models
To use your actual trained models, you have several options:

#### B1. Deploy Backend API (Recommended for Production)
```python
# Create a Flask API (app.py)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load your trained models
model = joblib.load('models/improved_marine_accident_classifier_model.pkl')
vectorizer = joblib.load('models/improved_marine_accident_classifier_vectorizer.pkl')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    text = data['description']
    
    # Use your actual model
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    severity_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    severity = severity_mapping[prediction]
    
    return jsonify({
        'severity': severity,
        'confidence': float(max(probabilities)),
        'probabilities': {
            'low': float(probabilities[0]),
            'medium': float(probabilities[1]),
            'high': float(probabilities[2])
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
```

#### B2. Deploy to Cloud Platforms
- **Heroku**: Free tier available
- **Railway**: Easy deployment
- **Render**: Free tier available
- **Vercel**: Great for Python APIs

#### B3. Update the Frontend
Replace the `simulateClassification` function in `index.html` with:

```javascript
async function classifyAccident() {
    const description = document.getElementById('accident-description').value.trim();
    
    if (!description) {
        alert('Please enter an accident description');
        return;
    }
    
    // Show loading state
    document.getElementById('results').style.display = 'block';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result-content').innerHTML = '';
    document.getElementById('classify-btn').disabled = true;
    
    try {
        const response = await fetch('https://your-api-url.com/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ description: description })
        });
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        alert('Error connecting to classification service');
        console.error('Error:', error);
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('classify-btn').disabled = false;
    }
}
```

## 📱 Features of the Interface

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Example Accidents**: Click to try predefined examples
- **Real-time Classification**: Instant results (when connected to API)
- **Visual Feedback**: Color-coded severity levels
- **Confidence Scores**: Shows model confidence
- **Probability Distribution**: Shows probabilities for all severity levels

## 🎨 Customization

### Change Colors
Edit the CSS variables in `index.html`:
```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --low-severity: #28a745;
    --medium-severity: #ffc107;
    --high-severity: #dc3545;
}
```

### Add More Examples
Edit the example section in the HTML:
```html
<div class="example-item" onclick="loadExample('Your new example here')">
    Your new example here
</div>
```

### Modify Styling
The interface uses modern CSS with:
- Gradient backgrounds
- Card-based layout
- Smooth animations
- Mobile-responsive design

## 🔒 Security Considerations

For production use:
1. **Rate Limiting**: Implement API rate limiting
2. **Input Validation**: Validate and sanitize user inputs
3. **CORS**: Configure CORS properly for your domain
4. **HTTPS**: Always use HTTPS in production

## 📊 Analytics (Optional)

Add Google Analytics to track usage:
```html
<!-- Add this in the <head> section -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

## 🚀 Next Steps

1. **Deploy the static version** (current `index.html`)
2. **Test the interface** with the examples
3. **Choose a backend option** if you want real ML predictions
4. **Customize the design** to match your brand
5. **Add analytics** to track usage

## 📞 Support

If you need help with deployment or customization:
1. Check GitHub Pages documentation
2. Review the Flask API example above
3. Consider using deployment platforms like Heroku or Railway

---

**Note**: The current version uses simulated results. For real ML predictions, implement one of the backend options described above. 