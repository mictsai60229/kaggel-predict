import os
import json
from .predict import predict_json, predict_batch_json


print("###testing###")
FILE_PATH = os.path.dirname(__file__)
TEST_FILE_NAME = os.path.join(FILE_PATH, "mc-examples.jsonl")
predict_json(json.load(open(TEST_FILE_NAME)))
predict_batch_json([json.loads(line) for line in open("mc-examples.jsonl")])
print("###testing done###")
