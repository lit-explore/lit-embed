import string
import numpy as np
from scipy.stats import poisson

def ridf(cf, df, N):
    """
    Computes the residual IDF (RIDF) of a given word.

    RIDF is a measure of the "informativeness" of a word in a corpus of documents.

    It uses a combination of IDF (which measures the _rareness_ of a word), and a
    Poisson model of expected word frequency.

    The Poisson model can be used to model the occurrence of common / non-content words.
    RIDF uses this to its advantage, by considering the deviation of a words occurrences
    from what is predicted by Poisson, in order to find words that are more
    interesting/informative.

    $$
    \text{RIDF} = \text{IDF} - \log_2 \frac{1}{1 - p(0; \lambda_i)}
    $$

    Parameters
    ----------
    cf: float
        the total number of occurrences of the word in the collection ("collection frequency")
    df: int
        the number of documents containing the word ("document frequency")
    N: int
        total number of documents

    References
    ----------
    1. Manning, C., & Schutze, H. (1999). Foundations of Statistical Natural Language
       Processing. MIT Press.
    """
    return np.log2(N / df) + np.log2(1 - poisson.pmf(0, cf / N))

def get_stop_words(lemmatize=False):
    """
    Returns a list of stop words
    """
    # source: gensim.parsing.preprocessing.STOPWORDS
    # modified to remove: "fire", "system"
    GENSIM_STOP_WORDS = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again',
            'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although',
            'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and',
            'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are',
            'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes',
            'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below',
            'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by',
            'call', 'can', 'cannot', 'cant', 'co', 'computer', 'con', 'could', 'couldnt',
            'cry', 'de', 'describe', 'detail', 'did', 'didn', 'do', 'does', 'doesn',
            'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight',
            'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even',
            'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few',
            'fifteen', 'fifty', 'fill', 'find', 'first', 'five', 'for', 'former',
            'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get',
            'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here',
            'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him',
            'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc',
            'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'kg',
            'km', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'make',
            'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover',
            'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely',
            'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none',
            'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on',
            'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our',
            'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please',
            'put', 'quite', 'rather', 're', 'really', 'regarding', 'same', 'say', 'see',
            'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should',
            'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow',
            'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such',
            'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves',
            'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
            'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those',
            'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together',
            'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under',
            'unless', 'until', 'up', 'upon', 'us', 'used', 'using', 'various', 'very',
            'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence',
            'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon',
            'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole',
            'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet',
            'you', 'your', 'yours', 'yourself', 'yourselves']

    # add common two-digit numbers
    STOP_WORDS = GENSIM_STOP_WORDS + [f"{x:02}" for x in range(101)]

    # add single letters and digits
    STOP_WORDS = STOP_WORDS + list(string.digits)
    STOP_WORDS = STOP_WORDS + list(string.ascii_lowercase)

    # add recent/upcoming years
    STOP_WORDS = STOP_WORDS + [f"{x:04}" for x in range(2000, 2030)]

    # other frequent but less informative words (subjective)
    STOP_WORDS = STOP_WORDS + ['ability', 'access', 'according', 'add', 'affect', 'aim',
            'aimed', 'aims', 'analysis', 'applications', 'approach', 'approaches',
            'article', 'assess', 'assessed', 'associated', 'based', 'better',
            'challenge', 'challenging', 'com', 'commonly', 'compared', 'consider',
            'corrigendum', 'current', 'currently', 'demonstrate', 'design', 'determine',
            'different', 'discuss', 'effect', 'effective', 'effects', 'erratum',
            'especially', 'evaluate', 'evidence', 'examined', 'existing', 'experience',
            'findings', 'focus', 'following', 'furthermore', 'given', 'healthy', 'http', 'https',
            'identified', 'important', 'including', 'induced', 'investigate',
            'investigated', 'key', 'known', 'like', 'main', 'mathbb', 'mathcal',
            'mathrm', 'method', 'methods', 'need', 'new', 'non', 'novel', 'observed',
            'obtain', 'obtained', 'org', 'outcome', 'overview', 'paper', 'particular',
            'performance', 'performed', 'possible', 'presence', 'present', 'problem',
            'problems', 'properties', 'propose', 'proposed', 'prove', 'provide',
            'range', 'recent', 'related', 'relationship', 'report', 'research',
            'respectively', 'result', 'results', 'revealed', 'review', 'role', 'set',
            'showed', 'shown', 'significant', 'significantly', 'solution', 'solutions',
            'studies', 'study', 'technique', 'test', 'theory', 'thing', 'try', 'type',
            'use', 'useful', 'value', 'values', 'way', 'work', 'www', 'year']

    # lemmatize?
    if lemmatize:
        import stanza
        from stanza.pipeline.core import ResourcesFileNotFoundError

        # create lemmatized version of stopwords
        try:
            nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
        except ResourcesFileNotFoundError:
            print("Downloading Stanza English language models for new install..")
            stanza.download('en')
            nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')

        doc = nlp(" ".join(STOP_WORDS))

        STOP_WORDS_LEMMA = []

        for sentence in doc.sentences:
            for word in sentence.words:
                STOP_WORDS_LEMMA.append(word.lemma)

        # work-around: "https" not properly removed?
        STOP_WORDS_LEMMA.append("https")

        STOP_WORDS_LEMMA = set(STOP_WORDS_LEMMA)

        return STOP_WORDS_LEMMA

    return STOP_WORDS
