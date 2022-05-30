from collections import Counter, OrderedDict
from torchtext.vocab import build_vocab_from_iterator, vocab
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")

def yield_tokens(sentences):
    ''' tokenize sentences
    
    Args:
        sentences: a list of sentence
            example:
            ['I love you', 'thank you']
        
    Returns:
        tokens:
            an iter of tokens, tokens is a list that split from sentence
            example:
            [['I', 'love', 'you'],
            ['thank', 'you']]
    '''
    for s in sentences:
        yield tokenizer(s)

def build_vocab(tokens):
    '''given the tokens, build a vocab
    Args:
        tokens: A 1-D list of token
    Returns:
        torchtext.vocab.vocab
    '''
    counter = Counter(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vcb = vocab(ordered_dict, specials=['[unk]', '[pad]'])
    # set_default_index给未知字符设置一个index
    vcb.set_default_index(0)
    return vcb


if __name__ == '__main__':
    pass