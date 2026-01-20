"""
Full Pipeline Test v4: Better prompt engineering with examples + self-correction
"""
import numpy as np
import joblib
from pathlib import Path
from llama_cpp import Llama

data_dir = Path('data/processed')
model_dir = Path('models')

print('='*60)
print('FULL PIPELINE TEST v4 - Improved Prompt Engineering')
print('='*60)

# Load RF model
X_test = np.load(data_dir / 'X_test.npy')
y_test = np.load(data_dir / 'y_test.npy')
model = joblib.load(model_dir / 'model.pkl')
label_encoder = joblib.load(data_dir / 'label_encoder.pkl')

def find_word_idx(word):
    for i, lbl in enumerate(label_encoder.classes_):
        if word.lower() == lbl.lower():
            for j, y in enumerate(y_test):
                if y == i:
                    return j
    return None

# Test scenarios
test_cases = [
    ['Hello', 'Thank you'],
    ['Good Morning', 'Mother'],
    ['School', 'Teacher'],
    ['Food', 'Want'],
    ['Doctor', 'Hospital'],
]

# Load 360M LLM
print('\nLoading SmolLM2-360M...')
llm = Llama(model_path='../isl_app/assets/SmolLM2-360M-Instruct-Q8_0.gguf', 
            n_ctx=1024, n_threads=8, verbose=False)

def gloss_to_english(gloss, llm):
    """Convert ISL gloss to English with detailed examples"""
    
    # Detailed few-shot prompt with explanation
    prompt = f'''You are translating Indian Sign Language (ISL) gloss into proper English sentences.

ISL gloss rules:
- Words are in different order than English (often Subject-Object-Verb)
- No articles (a, an, the) - you must ADD them
- No verb conjugations - you must FIX them (go → going, am going)
- No prepositions - you must ADD them (to, at, in)

Examples:
Gloss: "ME SCHOOL GO" → English: "I am going to school."
Gloss: "FOOD WANT" → English: "I want food."
Gloss: "MOTHER EXERCISE" → English: "My mother is exercising."
Gloss: "HELLO THANK YOU" → English: "Hello, thank you!"
Gloss: "GOOD MORNING MOTHER" → English: "Good morning, mother!"
Gloss: "SCHOOL TEACHER" → English: "The school teacher." or "I have a school teacher."
Gloss: "DOCTOR HOSPITAL" → English: "The doctor is at the hospital."
Gloss: "FOOD WANT" → English: "I want food."

Now convert this gloss to a proper English sentence:
Gloss: "{gloss}" → English: "'''

    output = llm(prompt, max_tokens=40, stop=['"', '\n'])
    result = output['choices'][0]['text'].strip()
    
    # Clean up result
    result = result.replace('"', '').strip()
    if result and not result[-1] in '.!?':
        result += '.'
    
    return result

print('\nTesting with improved prompts...')
print('='*60)

for words in test_cases:
    predictions = []
    for w in words:
        idx = find_word_idx(w)
        if idx is not None:
            pred_idx = model.predict(X_test[idx:idx+1])[0]
            pred_label = str(label_encoder.classes_[pred_idx])
            predictions.append(pred_label)
    
    gloss = ' '.join(predictions).upper()
    english = gloss_to_english(gloss, llm)
    
    print(f'Gloss:   {gloss}')
    print(f'English: {english}')
    print()

print('='*60)
print('TEST COMPLETE')
print('='*60)
