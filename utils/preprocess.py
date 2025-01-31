from janome.tokenizer import Tokenizer
#import MeCab

"""
Utilities for preprocessing sequence data.

Special tokens that are in all dictionaries:

<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<UNK>': 1,
  '<START>': 2,
  '<END>': 3,
}


def tokenize(s, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None):
  """
  Tokenize a sequence, converting a string s into a list of (string) tokens by
  splitting on the specified delimiter. Optionally keep or remove certain
  punctuation marks and add start and end tokens.
  """
  if punct_to_keep is not None:
    for p in punct_to_keep:
      s = s.replace(p, '%s%s' % (delim, p))

  if punct_to_remove is not None:
    for p in punct_to_remove:
      s = s.replace(p, '')

  tokens = s.split(delim)
  if add_start_token:
    tokens.insert(0, '<START>')
  if add_end_token:
    tokens.append('<END>')
  return tokens


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
  token_to_count = {}
  tokenize_kwargs = {
    'delim': delim,
    'punct_to_keep': punct_to_keep,
    'punct_to_remove': punct_to_remove,
  }
  for seq in sequences:
    seq_tokens = tokenize(seq, **tokenize_kwargs,
                    add_start_token=False, add_end_token=False)
    for token in seq_tokens:
      if token not in token_to_count:
        token_to_count[token] = 0
      token_to_count[token] += 1

  token_to_idx = {}
  for token, idx in SPECIAL_TOKENS.items():
    token_to_idx[token] = idx
  for token, count in sorted(token_to_count.items()):
    if count >= min_token_count:
      token_to_idx[token] = len(token_to_idx)

  return token_to_idx


def tokenize_jp(s, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None, tokenizer=None):
  """
  Tokenize a sequence, converting a string s into a list of (string) tokens by
  splitting on the specified delimiter. Optionally keep or remove certain
  punctuation marks and add start and end tokens.
  """

  #mecab = tokenizer.parse(s)
  
  #if punct_to_keep is not None:
    #for p in punct_to_keep:
      #s = s.replace(p, '%s%s' % (delim, p))

  if punct_to_remove is not None:
    for p in punct_to_remove:
      s = s.replace(p, '')

  words = tokenizer.tokenize(s, wakati = True)

  tokens = []
  for i in words:
      tokens.append(i)

 # tokens = s.split(delim)

  if add_start_token:
    tokens.insert(0, '<START>')
  if add_end_token:
    tokens.append('<END>')

  return tokens


def build_vocab_jp(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):

  token_to_count = {}
  tokenize_kwargs = {
    'delim': delim,
    'punct_to_keep': punct_to_keep,
    'punct_to_remove': punct_to_remove,
  }

  t = Tokenizer()

  for *_, seq, _ in sequences:
    seq_tokens = tokenize_jp(seq, **tokenize_kwargs,
                    add_start_token=False, add_end_token=False, tokenizer=t)

    for token in seq_tokens:
      if token not in token_to_count:
        token_to_count[token] = 0
      token_to_count[token] += 1

  token_to_idx = {}
  for token, idx in SPECIAL_TOKENS.items():
    token_to_idx[token] = idx
  for token, count in sorted(token_to_count.items()):
    if count >= min_token_count:
      token_to_idx[token] = len(token_to_idx)

  return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
  seq_idx = []
  for token in seq_tokens:
    if token not in token_to_idx:
      if allow_unk:
        token = '<UNK>'
      else:
        raise KeyError('Token "%s" not in vocab' % token)
    seq_idx.append(token_to_idx[token])
  return seq_idx


def encode_jp(seq_tokens, token_to_idx, allow_unk=False):
  seq_idx = []

  for token in seq_tokens:
    if token not in token_to_idx:
      if allow_unk:
        token = '<UNK>'
      else:
        raise KeyError('Token "%s" not in vocab' % token)
    seq_idx.append(token_to_idx[token])
  return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
  tokens = []
  for idx in seq_idx:
    tokens.append(idx_to_token[idx])
    if stop_at_end and tokens[-1] == '<END>':
      break
  if delim is None:
    return tokens
  else:
    return delim.join(tokens)


def encode_graph(graph_list, scene_to_idx, allow_unk=False):
  seq_idx = []

  for graph_i in graph_list:
    if not graph_i in scene_to_idx:
      if allow_unk:
        graph_i = "<UNK>"
      else:
        raise KeyError('Scene Graph "%s" not in dict' % flat_graph_i)
    seq_idx.append(scene_to_idx[graph_i])
  
  return seq_idx


def tokenize_graph(graph_data,
      add_start_token=True, 
      add_end_token=True
  ):

  graph_token = []
  for graph_i in graph_data:
    flat_graph_i = graph_i[0:1] + graph_i[1]
    graph_token.extend(flat_graph_i)

  if add_start_token:
    graph_token.insert(0, '<START>')
  if add_end_token:
    graph_token.append('<END>')

  return graph_token