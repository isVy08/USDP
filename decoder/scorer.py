import torch
import torch.nn.functional as F
from decoder.helper import *


def calSim(texts, evaluator):

    source = evaluator.encode([texts[-1]], convert_to_tensor=True)
    X = evaluator.encode(texts[:-1], convert_to_tensor=True)
    out = F.cosine_similarity(X, source)
    out = torch.clamp(out, min=0.0, max=1.0)
    return out

def calFluency(ordered_seq, critic):
    const = ['<s>'] + [token.pos_.lower() for token in ordered_seq if not isinstance(token, str)]
    n = len(const)
    f = 0
    for i in range(n-4):
        score = critic.logscore(const[i+4], [const[i:i+4]])
        f += score 
    return f/len(ordered_seq)


def score(prev_seq, token, input_text, evaluator, critic, tree):
    
    # order seq here
    
    ordered_seq = reorder(prev_seq, token)

    text = seq2text(ordered_seq)

    s = calSim([text, input_text], evaluator).item()
    f = calFluency(ordered_seq, critic)
    
    max_depth = 0
    for tok in ordered_seq:
        if not isinstance(tok, str):
            if tree[tok] > max_depth:
                max_depth = tree[tok]
    d = max_depth

    return s, f, d, text, ordered_seq


