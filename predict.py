from typing import List
import json
import torch
import os
import numpy as np
from allennlp.models.archival import load_archive
from allennlp.predictors import BidafPredictor, Predictor
from allennlp.common.checks import check_for_gpu
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.tokenizers import WordTokenizer
from .similarity import get_embedder


FILE_PATH = os.path.dirname(__file__)
PREDICT_MODEL_NAME = os.path.join(FILE_PATH, "model.tar.gz")
# chances are "fasttext", "stanford", "spacy"
# elmo will be added
SENTENCE_EMBEDDER_NAME = "fasttext"

sentence_embedder = get_embedder(SENTENCE_EMBEDDER_NAME)

def GetPredictor(archive: str) -> Predictor:
    check_for_gpu(1)
    archive = load_archive(archive, cuda_device=1)
    return Predictor.from_archive(archive)

predictor =  GetPredictor(PREDICT_MODEL_NAME)

'''
def GetElmo(sentences: List[str]):
    
    tokenizer = WordTokenizer()
    tokens = [[token.text for token in tokenizer.tokenize(sentence)] for sentence in sentences]
    
    return elmo_embedder.embed_batch(tokens)
'''
    

def torch_length(ndarrayA, ndarrayB):
    ndarrayA = np.linalg.norm(ndarrayA)
    ndarrayB = np.linalg.norm(ndarrayB)
    return np.linalg.norm(ndarrayA-ndarrayB)
    #return cosine(ndarrayA, ndarrayB)

def predict_batch_json(batch_json_data: List[dict], batch_size: int = 4):
    global predictor
    results = [json_data['best_span_str']
              for idx in range(0, len(batch_json_data) ,batch_size)
              for json_data in predictor.predict_batch_json(batch_json_data[idx: idx+batch_size])]
    #results = [json_data['best_span_str'] for json_data in predictor.predict_batch_json(batch_json_data)]
    
    batch_result = []
    for result, json_data in zip(results, batch_json_data):
        sentences = [result]
        sentences_idx = ["predict"]
        for option_idx, option in json_data["options"]:
            sentences_idx.append(option_idx)
            sentences.append(option)
        
        tensors = {sentence_idx: ndarray for sentence_idx, ndarray in zip(sentences_idx, sentence_embedder.encode(sentences) )}

        distances = {sentence_idx: sentence_embedder.similarity(tensors['predict'] ,tensors[sentence_idx]) for sentence_idx in sentences_idx[1:]}
        
        batch_result.append({"predict": result, "cosine": distances})
        
    return batch_result

def predict_json(json_data: dict):
    global predictor
    result = predictor.predict_json(json_data)['best_span_str']
    
    sentences = [result]
    sentences_idx = ["predict"]
    for option_idx, option in json_data["options"]:
        sentences_idx.append(option_idx)
        sentences.append(option)
    
        
    tensors = {sentence_idx: ndarray for sentence_idx, ndarray in zip(sentences_idx, sentence_embedder.encode(sentences) )}

    distances = {sentence_idx: sentence_embedder.similarity(tensors['predict'] ,tensors[sentence_idx]) for sentence_idx in sentences_idx[1:]}
    
    return {"predict": result, "cosine": distances}
    

def predict(question: str, passage: str, options: List[str]):
    return predict_json({'question': question, "passage": passage, "options": options})
    
