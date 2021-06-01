
import msgpackrpc

from .types import *
from .sensor import *

class RadarSensor(Sensor):
    def __init__(self, client, ID):
        super(RadarSensor, self).__init__(client, ID, ESensorType().Radar)
    
    def get_sensor_detections(self):
        print("RadarSensor::get_sensor_detections is deprecated, will be removed in the next release. Please use Sensor::get_detections!")
        return self.client.call("GetRadarSensorDetections", self._ID)
