import torch
import numpy as np
import spacy
from .models import InferSent
import os

FILE_PATH = os.path.dirname(__file__)
FASTEXT_MODEL_PATH = os.path.join(FILE_PATH, 'encoder/infersent2.pkl')
FASTTEXT_W2V_PATH = os.path.join(FILE_PATH, 'dataset/fastText/crawl-300d-2M.vec')
STANFORD_MODEL_PATH = os.path.join(FILE_PATH, 'encoder/infersent1.pkl')
STANFORD_W2V_PATH = os.path.join(FILE_PATH, 'dataset/GloVe/glove.840B.300d.txt')
def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

def get_embedder(embedder_type="fasttext"):
    embedder_name = {
        'fasttext' : fasttext_embedder,
        'stanford' : stanford_embedder,
        'spacy'    : spacy_embedder  
    }
    embedder = embedder_name[embedder_type]
    
    return embedder()
    

class fasttext_embedder(object):
    
    def __init__(self):
        #facebook inference
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
        self.model = InferSent(params_model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(FASTEXT_MODEL_PATH))
        

        self.model.set_w2v_path(FASTTEXT_W2V_PATH)
        self.model.build_vocab_k_words(K=500000)
    
    def encode(self, sentences):
         return self.model.encode(sentences, bsize=128, tokenize=False, verbose=True)
    
    def similarity(self, ndarrayA, ndarrayB):
        return cosine(ndarrayA, ndarrayB)

class stanford_embedder(object):
    
    def __init__(self):
        
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        self.model = InferSent(params_model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(STANFORD_MODEL_PATH))
        
        self.model.set_w2v_path(STANFORD_W2V_PATH)
        self.model.build_vocab_k_words(K=500000)
    
    def encode(self, sentences):
         return self.model.encode(sentences, bsize=128, tokenize=False, verbose=True)
    
    def similarity(self, ndarrayA, ndarrayB):
        return cosine(ndarrayA, ndarrayB)
    
    
class spacy_embedder(object):
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        
    def encode(self, sentences):
        return [self. nlp(sentence).vector for sentence in sentences]
    
    def similarity(self, ndarrayA, ndarrayB):
        return cosine(ndarrayA, ndarrayB)
