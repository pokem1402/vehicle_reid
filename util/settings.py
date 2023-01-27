import os
# class ReID:
#     MODEL_PATH = "models/vehicleID_.onnx"
#     SIZE = (256, 256)   # width, height
class ReID:
    SIZE = (256, 256)   # width, height
    def __init__(self, model_name):
        self.MODEL_PATH = "models/"+model_name
        if not os.path.exists(self.MODEL_PATH):
            raise ValueError
