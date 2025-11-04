"""
Query preprocessing utilities
Apply same preprocessing as used during indexing
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List


class QueryPreprocessor:
    """Preprocess queries the same way as documents were preprocessed"""
    
    def __init__(self):
        # Download NLTK resources if needed
        self._download_nltk_resources()
        
        # Initialize stemmer and stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        resources = ['punkt', 'stopwords', 'punkt_tab']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query text the same way as documents
        Returns space-separated stemmed tokens
        """
        # Convert to lowercase
        text = query.lower()
        
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text)
        
        # Remove punctuation (except spaces)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Filter: remove numbers, short tokens
        tokens = [token for token in tokens if not token.isdigit() and len(token) >= 2]
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stem tokens
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Return as space-separated string
        return ' '.join(tokens)
    
    def preprocess_query_list(self, query: str) -> List[str]:
        """
        Preprocess query and return as list of tokens
        """
        processed = self.preprocess_query(query)
        return processed.split() if processed else []


# Global instance for reuse
_query_preprocessor = None

def get_query_preprocessor() -> QueryPreprocessor:
    """Get or create global query preprocessor instance"""
    global _query_preprocessor
    if _query_preprocessor is None:
        _query_preprocessor = QueryPreprocessor()
    return _query_preprocessor


def preprocess_query(query: str) -> str:
    """Convenience function to preprocess a query"""
    return get_query_preprocessor().preprocess_query(query)


def preprocess_query_list(query: str) -> List[str]:
    """Convenience function to preprocess a query and return tokens"""
    return get_query_preprocessor().preprocess_query_list(query)
