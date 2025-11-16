import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = [
        stemmer.stem(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(words)

if __name__ == "__main__":
    exemplo = "Olá! Preciso que você prepare o relatório mensal. Obrigado!"
    resultado = preprocess_text(exemplo)
    print(f"Original: {exemplo}")
    print(f"Processado: {resultado}")
