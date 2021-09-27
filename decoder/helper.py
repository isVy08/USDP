def inverse_dict(d):
    inv_d = {}
    for k, v in d.items():
        if v not in inv_d: 
            inv_d[v] = []
        inv_d[v].append(k)
    return inv_d 


def get_parent_child(doc):
    c2p = {}
    for token in doc:
        c2p[token] = token.head
    p2c = inverse_dict(c2p)
    return p2c

def get_relations(doc, subj):
    
    lookup = {}
    for token in doc:
        children = list(token.children) if not isinstance(token.children, list) else token.children
        if token.i == subj.i:
            lookup[token] = [token.head] + children
        
        else:
            lookup[token] = children

    return lookup

def get_root_head(doc, lang):
    if lang == 'en':
        for N in list(doc.noun_chunks): 
            if N[-1].head.dep_ == 'ROOT':
                return list(N)
    return [doc[0]]

def reorder(prev_seq, token):

    if isinstance(token, str):
        return prev_seq + [token]
    
    i = 0
    while i < len(prev_seq): 
        seq_token = prev_seq[i]
        if not isinstance(seq_token, str) and seq_token.i > token.i:
            return prev_seq[:i] + [token] + prev_seq[i:]
        i += 1
    return prev_seq + [token]


def seq2text(ordered_seq):
    
    sents = []
    for token in ordered_seq:
        if isinstance(token, str):
            sents.append('<s>')
        else: 
            sents.append(token.text)
    return ' '.join(sents)
    

def get_tree_depth(doc):
    tree = {}
    tree_depth = 0
    for token in doc:
        ans = list(token.ancestors) if not isinstance(token.ancestors, list) else token.ancestors
        depth = len(ans) + 1
        tree[token] = depth
        if depth > tree_depth:
            tree_depth = depth
    return tree, tree_depth

def split(ordered_seq):
    sent = ''
    branch = []
    started = False
    for token in ordered_seq:
        if isinstance(token, str):
            if started:                    
                text = '- <mask> ' + seq2text(branch) + ' '
            else: 
                started = True
                text = seq2text(branch) + ' '
            sent += text
            branch = []
        else: 
            branch.append(token)

    return sent
