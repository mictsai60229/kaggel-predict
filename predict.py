from typing import List
import json
import torch
import numpy as np
import spacy
from allennlp.models.archival import load_archive
from allennlp.predictors import BidafPredictor, Predictor
from allennlp.common.checks import check_for_gpu
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.tokenizers import WordTokenizer

PREDICT_MODEL_NAME = "model.tar.gz"
nlp = spacy.load('en_core_web_sm')
#elmo_embedder = ElmoEmbedder()

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

def predict_batch_json(batch_json_data: List[dict]):
    global predictor
    results = [predictor.predict_json(json_data)['best_span_str'] for json_data in batch_json_data]
    
    batch_result = []
    for result, json_data in zip(results, batch_json_data):
        sentences = [result]
        sentences_idx = ["predict"]
        for option_idx, option in json_data["options"]:
            sentences_idx.append(option_idx)
            sentences.append(option)
            
        #elmo = GetElmo(sentences = sentences)
        tensors = {sentence_idx: nlp(sentence) for sentence_idx, sentence in zip(sentences_idx, sentences)}

        distances = {sentence_idx: tensors['predict'].similarity(tensors[sentence_idx]) for sentence_idx in sentences_idx[1:]}

        return {"predict": result, "cosine": distances}
        
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
    
    #elmo = GetElmo(sentences = sentences)
    tensors = {sentence_idx: nlp(sentence) for sentence_idx, sentence in zip(sentences_idx, sentences)}
    
    distances = {sentence_idx: tensors['predict'].similarity(tensors[sentence_idx]) for sentence_idx in sentences_idx[1:]}
    
    return {"predict": result, "cosine": distances}
    

def predict(question: str, passage: str, options: List[str]):
    return predict_json({'question': question, "passage": passage, "options": options})
    


if __name__ == "__main__":
    json_data = [json.loads(line) for line in open("mc-examples.jsonl") if line.strip()]
    print(predict_batch_json(json_data))
    
    
    json_data = json.load(open("mc-examples.jsonl"))
    print(predict_json(json_data))
    
    predict(json_data['question'] ,json_data["passage"], json_data["options"])