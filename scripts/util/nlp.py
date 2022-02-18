import stanza
from stanza.pipeline.core import ResourcesFileNotFoundError

# source: gensim.parsing.preprocessing.STOPWORDS
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
        'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former',
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
        'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves',
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

STOP_WORDS = GENSIM_STOP_WORDS + ['10', 'analysis', 'applications', 'approach',
        'associated', 'based', 'case', 'compared', 'consider', 'current', 'data',
        'demonstrate', 'different', 'effect', 'effective', 'effects', 'equation',
        'equations', 'evidence', 'examined', 'findings', 'following', 'function',
        'functions', 'given', 'identified', 'important', 'including', 'induced',
        'investigate', 'investigated', 'known', 'large', 'like', 'mathbb', 'mathcal',
        'mathrm', 'method', 'methods', 'model', 'models', 'new', 'non', 'novel',
        'observed', 'obtain', 'obtained', 'paper', 'parameter', 'parameters',
        'particular', 'performance', 'performed', 'possible', 'presence', 'present',
        'problem', 'problems', 'properties', 'propose', 'proposed', 'prove', 'provide',
        'range', 'recent', 'related', 'respectively', 'result', 'results', 'revealed',
        'set', 'showed', 'shown', 'significant', 'significantly', 'solution',
        'solutions', 'studies', 'study', 'term', 'terms', 'theory', 'type', 'use',
        'values', 'work']

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

STOP_WORDS_LEMMA = set(STOP_WORDS_LEMMA)
