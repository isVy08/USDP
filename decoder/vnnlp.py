from vncorenlp import VnCoreNLP

def pos_converter(pos):
    """Convert common VN pos tags to EN spacy equivalences"""
    cv = {
        'A': 'ADJ',
        'V': 'VERB',
        'N': 'NOUN',
        'L': 'DET',
        'Np': 'PROPN',
        'P': 'PRON',
        'T': 'PART',
        'CN': 'PUNCT',
        'M': 'NUM',
        'R': 'ADV',
        'E': 'ADP',
        'Cc': 'CCONJ',
        'C': 'SCONJ'
        }
    try: 
        return cv[pos]
    except KeyError:
        return pos.upper()

def dep_converter(dep):
    cv = {'sub':'nsubj', 'dob':'dobj', 'iob': 'iobj', 'root':'ROOT'}
    try:
        return cv[dep]
    except KeyError:
        return dep


class Token:
    def __init__(self, index, text, pos, dep, head_index, doc):
        self.i = index - 1
        self.pos_ = pos_converter(pos) 
        self.dep_ = dep_converter(dep)
        self.text = text
        self.head_index = head_index - 1 
        self.doc = doc
        self.lemma_ = None # tbu
        self.tag_ = None # tbu
        self.is_root = True if self.dep_ == 'ROOT' else False
    
    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def get_head(self):
        self.head = self.doc[self.head_index]
        
    def get_ancestors(self):
        self.ancestors = []
        if not self.is_root:
            head = self.head            
            while not head.is_root and head not in self.ancestors:
                self.ancestors.append(head)
                head = head.head
            if head not in self.ancestors:
                self.ancestors.append(head)
    
    def get_children(self):
        self.children = []
        for token in self.doc:
            if token.head_index == self.i and not token.is_root:
                self.children.append(token)
    

class nlp:
    def __init__(self, sentences):
        
        # parse text 
        annotator = VnCoreNLP('VnCoreNLP-1.1.1.jar')
        outputs = annotator.annotate(sentences)
        annotator.close()
        self._doc = list()
        self.index = 0
        self.text = sentences
        self.root = None

        # parse each sentence
        i = 0
        for sent in outputs['sentences']:
            for token in sent:
                token_index = token['index'] + i
                if token['depLabel'] == 'root': 
                    tok = Token(token_index, token['form'], token['posTag'], token['depLabel'], token_index, self) 
                    if self.root is None:
                        self.root = tok
                else: 
                    token_head_index = token['head'] + i 
                    tok = Token(token_index, token['form'], token['posTag'], token['depLabel'], token_head_index, self) 
                self._doc.append(tok)
            i = token_index

        # generate family 
        self.activate()
            
    
    def activate(self):
        """Create token head"""
        for token in self._doc:
            if self.root is None and token.dep_[0] in ('v', 'V'):
                self.root = token
                token.dep_ = 'ROOT'
                token.is_root = True
            token.get_head()
        
        for token in self._doc:
            token.get_ancestors()
            token.get_children()
            
    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self._doc.__len__()

    def __iter__(self):
        return self
    
    def __next__(self):
        try: 
            result = self._doc[self.index]
        except IndexError:
            self.index = 0
            raise StopIteration
        
        self.index +=1
        return result
               
    def __getitem__(self, index):
        return self._doc[index]
    




