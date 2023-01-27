
from util.util import sample_distance
from util.settings import ReID

if __name__ == "__main__":

    vehicleID = ReID("vehicleID_.onnx")
    
    sample_distance(vehicleID)


