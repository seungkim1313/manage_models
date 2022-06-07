import numpy as np
import bentoml
from bentoml.io import JSON
import torch

model_runner = bentoml.pytorch.load_runner("cyclegan_vangogh:latest", predict_fn_name='forward')

v2_service = bentoml.Service("style_transfer_vangogh", runners=[model_runner])

@v2_service.api(input=JSON(), output=JSON())
def mobilenet_classify(parsed_json):
    print(parsed_json)
    d = parsed_json.encode('latin')
    print(d, type(d))

    import pickle

    hi = pickle.loads(d)
    print(hi)

    #data = {"A": None, "A_paths": None}
    #data['A'] = torch.FloatTensor(input_series)
    #result = model_runner.run(torch.FloatTensor(input_series))