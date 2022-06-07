import cv2
import torch
import numpy as np


def prepare_input(image, config):
    model_input = {"A": None, "A_paths": None}

    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array([image])
    image = image.transpose([0, 3, 1, 2])

    model_input['A'] = torch.FloatTensor(image)

    return model_input


#def prepare_output(image, config):


class CustomModel(torch.nn.Module):
    
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = create_model(config)
        self.model.setup(config)

    def forward(self, **kwargs):
        #image = kwargs['image'].to(torch.uint8).numpy()[0]
        image = kwargs['image']
        data = prepare_input(image, config)
        self.model.set_input(data)
        self.model.test()
        model_result = self.model.get_current_visuals()['fake']
        result_image = prepare_output(model_result, config)
        return result_image
    