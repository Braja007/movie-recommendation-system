import re

# Basic stopword list (manually added)
stop_words = {
    "a", "an", "the", "is", "in", "on", "and", "or", "if", "of", "to", "with", "as",
    "for", "was", "were", "by", "this", "that", "it", "at", "from", "but", "be", "are",
    "have", "has", "had", "not", "can", "will", "just", "they", "them", "he", "she",
    "you", "we", "i", "do", "does", "did", "so", "than", "too"
}

def preprocess_text(text):
    """
    Cleans and tokenizes input text:
    - Removes non-alphabet characters
    - Converts to lowercase
    - Removes stopwords
    """
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))  # Remove non-alphabetic
    text = text.lower()
    tokens = text.split()  # Simple whitespace tokenization
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
