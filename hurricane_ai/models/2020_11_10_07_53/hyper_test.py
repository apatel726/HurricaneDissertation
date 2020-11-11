import json
import pprint
with open('hyperparameters.json') as f:
            root = json.load(f)
name = root["config"]        
featuresearch = name["name"]
#print(featuresearch)

if featuresearch == "sequential":
    model = "wind"
elif featuresearch == 'sequential_1':
    model = "lat"
else:
    model = "long"
print(model)
