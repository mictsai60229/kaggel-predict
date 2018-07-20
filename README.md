README
===========================

# pytorch >= 0.4

請先執行download.sh
```
sh download.sh
```

接著確認nltk module
```python
import nltk
nltk.download('punkt')
```

To test it's fine
```shell
python -m <download-repo-name>
```
How to use it?
```python
from <download-repo-name> import predict_json

json_data = {"passage": "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.", "question": "How many partially reusable launch systems were developed?", "options": [["1","1"], ["2","2"], ["3", "3"], ["4", "4"]]}

print(predict_json(json_data)) 

```


