
import msgpackrpc

from .types import *
from .sensor import *

class DistanceSensor(Sensor):
    def __init__(self, client, ID):
        super(DistanceSensor, self).__init__(client, ID, ESensorType().Distance)
    
    def get_sensor_detections(self):
        print("DistanceSensor::get_sensor_detections is deprecated, will be removed in the next release. Please use Sensor::get_detections!")
        return self.client.call("GetDistanceSensorDetections", self._ID)
