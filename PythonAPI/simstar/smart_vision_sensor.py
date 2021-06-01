import msgpackrpc

from .types import *
from .sensor import *

class SmartVisionSensor(Sensor):
    def __init__(self, client, ID):
        super(SmartVisionSensor, self).__init__(client, ID, ESensorType().Vision)
    
    def get_sensor_detections(self):
        print("SmartVisionSensor::get_sensor_detections is deprecated, will be removed in the next release. Please use Sensor::get_detections!")
        return self.client.call("GetSmartVisionSensorDetections", self._ID)

    def get_lane_points(self):
        print("SmartVisionSensor::get_lane_points is deprecated, will be removed in the next release. Please use Sensor::get_lane_detections!")
        return self.client.call("GetVisionLanePoints", self._ID)

    def get_annotations(self):
        return self.client.call("GetVisionAnnotations",self._ID)