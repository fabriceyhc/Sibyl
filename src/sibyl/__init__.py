import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from .config import *
from .transforms import *
from .utils import *