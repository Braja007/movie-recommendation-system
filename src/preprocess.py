import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Create a local nltk_data folder
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

# Set the NLTK data path
nltk.data.path.append(nltk_data_path)

# Download required NLTK resources to the local folder
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# Set of English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and tokenizes input text.
    - Removes non-alphabet characters
    - Converts to lowercase
    - Removes stopwords
    """
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
