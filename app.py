from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import PyPDF2
import docx
import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import magic

# Download NLTK data (run this once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        raise Exception(f"Error reading TXT file: {str(e)}")

def analyze_readability(text):
    """Analyze text readability metrics"""
    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'smog_index': textstat.smog_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text)
    }

def analyze_structure(text):
    """Analyze document structure"""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Calculate average sentence length
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    
    # Count paragraphs (approximate by double newlines)
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'avg_words_per_paragraph': round(len(words) / len(paragraphs), 2) if paragraphs else 0
    }

def detect_passive_voice(text):
    """Simple passive voice detection"""
    passive_patterns = [
        r'\b(am|is|are|was|were|be|being|been)\s+\w+ed\b',
        r'\b(am|is|are|was|were|be|being|been)\s+\w+en\b',
    ]
    
    passive_count = 0
    sentences = sent_tokenize(text)
    
    for sentence in sentences:
        for pattern in passive_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                passive_count += 1
                break
    
    return {
        'passive_sentences': passive_count,
        'passive_percentage': round((passive_count / len(sentences)) * 100, 2) if sentences else 0
    }

def check_academic_tone(text):
    """Check for academic writing style indicators"""
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords and punctuation
    content_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Simple academic word list (you can expand this)
    academic_words = {
        'however', 'therefore', 'moreover', 'furthermore', 'consequently',
        'nevertheless', 'notwithstanding', 'accordingly', 'additionally',
        'significantly', 'substantially', 'considerably', 'remarkably'
    }
    
    academic_word_count = len([word for word in content_words if word in academic_words])
    academic_word_ratio = round((academic_word_count / len(content_words)) * 100, 2) if content_words else 0
    
    return {
        'academic_word_count': academic_word_count,
        'academic_word_ratio': academic_word_ratio
    }

def check_spelling_grammar_issues(text):
    """Basic spelling and grammar checks"""
    issues = []
    
    # Check for common issues
    sentences = sent_tokenize(text)
    
    for i, sentence in enumerate(sentences):
        # Check sentence length
        if len(word_tokenize(sentence)) > 50:
            issues.append({
                'type': 'long_sentence',
                'description': 'Sentence is very long and may be hard to read',
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'severity': 'medium'
            })
        
        # Check for repetitive words
        words = word_tokenize(sentence.lower())
        word_freq = {}
        for word in words:
            if word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repetitive_words = [word for word, count in word_freq.items() if count > 3]
        if repetitive_words:
            issues.append({
                'type': 'word_repetition',
                'description': f'Repeated words: {", ".join(repetitive_words[:3])}',
                'sentence': sentence,
                'severity': 'low'
            })
    
    return issues

def generate_recommendations(analysis):
    """Generate recommendations based on analysis"""
    recommendations = []
    readability = analysis['readability']
    structure = analysis['structure']
    passive_voice = analysis['passive_voice']
    academic_tone = analysis['academic_tone']
    
    # Readability recommendations
    if readability['flesch_reading_ease'] < 30:
        recommendations.append({
            'category': 'Readability',
            'title': 'Very Difficult Text',
            'description': 'The text is very difficult to read. Consider simplifying sentence structures and vocabulary.',
            'severity': 'high',
            'suggestion': 'Aim for a Flesch Reading Ease score between 30-50 for academic writing.'
        })
    
    if readability['flesch_kincaid_grade'] > 16:
        recommendations.append({
            'category': 'Readability',
            'title': 'High Education Level Required',
            'description': f"The text requires {readability['flesch_kincaid_grade']:.1f} years of education to understand.",
            'severity': 'medium',
            'suggestion': 'Consider making the text more accessible without losing academic rigor.'
        })
    
    # Structure recommendations
    if structure['avg_sentence_length'] > 25:
        recommendations.append({
            'category': 'Structure',
            'title': 'Long Sentences',
            'description': 'Average sentence length is quite long, which may affect readability.',
            'severity': 'medium',
            'suggestion': 'Break long sentences into shorter, more focused ones.'
        })
    
    # Passive voice recommendations
    if passive_voice['passive_percentage'] > 20:
        recommendations.append({
            'category': 'Writing Style',
            'title': 'Passive Voice Overuse',
            'description': f"{passive_voice['passive_percentage']}% of sentences use passive voice.",
            'severity': 'medium',
            'suggestion': 'Use active voice for more direct and engaging writing.'
        })
    
    # Academic tone recommendations
    if academic_tone['academic_word_ratio'] < 5:
        recommendations.append({
            'category': 'Academic Style',
            'title': 'Limited Academic Vocabulary',
            'description': 'The text could benefit from more academic transition words and phrases.',
            'severity': 'low',
            'suggestion': 'Incorporate academic transition words like "however", "therefore", "moreover".'
        })
    
    # Add spelling/grammar issues
    for issue in analysis['spelling_grammar_issues']:
        recommendations.append({
            'category': 'Grammar & Style',
            'title': issue['type'].replace('_', ' ').title(),
            'description': issue['description'],
            'severity': issue['severity'],
            'suggestion': 'Review the highlighted sentence for improvement.'
        })
    
    return recommendations

def calculate_overall_score(analysis):
    """Calculate an overall score based on various metrics"""
    score = 100
    
    # Deduct for readability issues
    readability = analysis['readability']
    if readability['flesch_reading_ease'] < 20:
        score -= 20
    elif readability['flesch_reading_ease'] < 30:
        score -= 10
    
    # Deduct for structure issues
    structure = analysis['structure']
    if structure['avg_sentence_length'] > 30:
        score -= 15
    elif structure['avg_sentence_length'] > 25:
        score -= 8
    
    # Deduct for passive voice
    passive_voice = analysis['passive_voice']
    if passive_voice['passive_percentage'] > 25:
        score -= 15
    elif passive_voice['passive_percentage'] > 15:
        score -= 8
    
    # Ensure score is between 0-100
    return max(0, min(100, round(score)))

@app.route('/api/analyze-thesis', methods=['POST'])
def analyze_thesis():
    """Main endpoint for thesis analysis"""
    try:
        # Check if file was uploaded
        if 'thesis' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['thesis']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload PDF, DOC, DOCX, or TXT files.'}), 400
        
        # Save the file temporarily
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)
        
        # Extract text based on file type
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension in ['doc', 'docx']:
            text = extract_text_from_docx(file_path)
        else:  # txt
            text = extract_text_from_txt(file_path)
        
        # Check if text was extracted successfully
        if not text or len(text.strip()) < 100:
            return jsonify({'error': 'Unable to extract sufficient text from the file. The file may be empty, corrupted, or contain only images.'}), 400
        
        # Perform analysis
        readability = analyze_readability(text)
        structure = analyze_structure(text)
        passive_voice = detect_passive_voice(text)
        academic_tone = check_academic_tone(text)
        spelling_grammar_issues = check_spelling_grammar_issues(text)
        
        # Combine analysis results
        analysis_results = {
            'readability': readability,
            'structure': structure,
            'passive_voice': passive_voice,
            'academic_tone': academic_tone,
            'spelling_grammar_issues': spelling_grammar_issues
        }
        
        # Generate recommendations and overall score
        recommendations = generate_recommendations(analysis_results)
        overall_score = calculate_overall_score(analysis_results)
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'overallScore': overall_score,
            'statistics': {
                'wordCount': structure['word_count'],
                'sentenceCount': structure['sentence_count'],
                'paragraphCount': structure['paragraph_count'],
                'readabilityScore': readability['flesch_reading_ease']
            },
            'recommendations': recommendations
        })
        
    except Exception as e:
        # Clean up in case of error
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Thesis analyzer API is running'})

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    print("Starting Thesis Analyzer API...")
    print("API will be available at: http://localhost:5000")
    print("Endpoints:")
    print("  GET  /api/health")
    print("  POST /api/analyze-thesis")
    
    app.run(debug=True, host='0.0.0.0', port=5000)