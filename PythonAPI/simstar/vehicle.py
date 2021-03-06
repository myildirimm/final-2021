

import math
import msgpackrpc

from .types import *
from .distance_sensor import *
from .radar_sensor import *
from .imu_sensor import *
from .smart_vision_sensor import *

class Vehicle():
    def __init__(self,client,ID,timeout=10):
        self._ID = int(ID)
        self.local_client = client
        server_info = self.local_client.call("GetServerNetInfo")
        if server_info == "":
            self.client = client
        else:
            ip_port = server_info.split(":")
            self.client = msgpackrpc.Client(msgpackrpc.Address(ip_port[0], int(ip_port[1])), timeout = timeout, \
                pack_encoding = 'utf-8', unpack_encoding = 'utf-8')

    def get_ID(self):
        return self._ID
    
    def set_controller_type(self, controller_type):
        self.client.call("SetControllerType", self._ID, controller_type)
    
    def set_custom_vehicle_model(self, vehicle_model_type):
        self.client.call("SetCustomVehicleModel", self._ID, vehicle_model_type)

    def control_vehicle(self,throttle,steer,brake):
        control = VehicleControl(throttle,steer,brake)
        self.client.call("ControlVehicle",control,self._ID)

    def get_sensor_method(self, sensor_type):
        return {
            ESensorType().Camera: "AddCameraSensor",
            ESensorType().Radar: "AddRadarSensor",
            ESensorType().Lidar: "AddLidar",
            ESensorType().Vision: "AddSmartVisionSensor",
            ESensorType().Distance: "AddDistanceSensor",
            ESensorType().Imu: "AddIMUSensor",
            ESensorType().Gnss: "AddGNSSSensor",
            ESensorType().Encoder: "AddEncoderSensor",
            ESensorType().RoadSurface:"AddRoadSurfaceSensor",
            ESensorType().SemanticCamera:"AddSemanticSegmentationCamera",
        }[sensor_type]

    def add_sensor(self, sensor_type, sensor_parameters=None):
        if(sensor_parameters):
            sensor_id = self.local_client.call(self.get_sensor_method(sensor_type), self._ID, sensor_parameters)
        else:
            # Used to add imu/encoder/road_surface sensor with default sensor name
            sensor_id = self.local_client.call(self.get_sensor_method(sensor_type), self._ID, "")
        return Sensor(self.local_client, sensor_id, sensor_type)
        
    def add_sensor_from_json(self, json_data):
        return self.client.call("AddSensorsFromJson", self._ID, json_data)

    def get_attached_sensor(self, sensor_name):
        sensors = self.local_client.call("GetAttachedSensor", self._ID, sensor_name)
        attached_sensors = []
        for sensor in sensors:
            attached_sensors.append(Sensor(self.local_client, sensor['ID'], sensor['sensor_type']))
        return attached_sensors

    def add_distance_sensor(self, sensor_parameters):
        print("add_distance_sensor is deprecated, will be removed in the next release. Please use add_sensor!")
        sensor_id = self.client.call("AddDistanceSensor", self._ID, sensor_parameters)
        return DistanceSensor(self.client, sensor_id)
        
    def add_radar_sensor(self, sensor_parameters):
        print("add_radar_sensor is deprecated, will be removed in the next release. Please use add_sensor!")
        sensor_id = self.client.call("AddRadarSensor", self._ID, sensor_parameters)
        return RadarSensor(self.client, sensor_id)

    def add_imu_sensor(self):
        print("add_imu_sensor is deprecated, will be removed in the next release. Please use add_sensor!")
        sensor_id = self.client.call("AddIMUSensor", self._ID, "")
        return ImuSensor(self.client, sensor_id)
        
    def add_smart_vision_sensor(self, sensor_parameters):
        print("add_smart_vision_sensor is deprecated, will be removed in the next release. Please use add_sensor!")
        sensor_id = self.client.call("AddSmartVisionSensor", self._ID, sensor_parameters)
        return SmartVisionSensor(self.client, sensor_id)

    def get_speed(self):
        vehicle_state = self.get_vehicle_state_self_frame()
        velocity_x_ms = abs( float(vehicle_state['velocity']['X_v']) )
        velocity_y_ms = abs( float(vehicle_state['velocity']['Y_v']) )
        speed = math.sqrt(
            math.pow(velocity_x_ms, 2) +
            math.pow(velocity_y_ms, 2))
        return speed

    # global frame
    def get_vehicle_state(self):
        return self.client.call("GetVehicleStateInfo",self._ID)

    # in vehicle frame
    def get_vehicle_state_self_frame(self):
        return self.client.call("GetVehicleStateInfoInSelfFrame",self._ID)

    def check_for_collision(self):
        return self.client.call("GetVehicleInfoIsCollided",self._ID)

    def get_road_deviation_info(self):
        return self.client.call("GetRoadDeviationInfo",self._ID)

    def get_controller_inputs(self):
        return self.client.call("GetVehicleControlInputs",self._ID)

    def set_lane_assist(self, assist_type):
        self.client.call("SetLaneAssist", self._ID, assist_type)
        
    def set_cruise_control(self, cruise_control):
        self.client.call("SetCruiseControl", self._ID, cruise_control)
    
    def set_waypoint(self, trajectory):
        self.client.call("SetWaypoint", self._ID, trajectory)
        
    def set_driver_profile(self, profile):
        self.client.call("SetDriverProfile", self._ID, profile)

    def set_driver_highway_exit_chance(self, chance=0):
        self.client.call("SetHighwayExitChance", self._ID, chance)

    def order_lane_change(self, direction):
        self.client.call("ChangeLane", self._ID, direction)

    def get_physical_properties(self):
        return self.client.call("GetVehiclePhysicalProperties", self._ID)

    def set_physical_properties(self, vehicle_setup):
        self.client.call("SetVehiclePhysicalProperties", self._ID, vehicle_setup)

    def assign_relative_trajectory(self, trajectory):
        self.client.call("AssignRelativeTrajectory", self._ID, trajectory)
    
    def get_attached_sensors(self):
        sensor_list = self.client.call("FindAttachedSensor",self._ID)
        return sensor_list

    def assign_intersection_trajectory(self, trajectory):
        self.client.call("AssignIntersectionTrajectory", self._ID, trajectory)

    def is_lane_changing(self):
        return self.client.call("GetVehicleInfoIsLaneChanging", self._ID)

    def set_target_way_point(self, target_location, track_lanes=True):
        return self.client.call("SetTargetWayPoint", self._ID, target_location, track_lanes)
        
    def get_path_to_target_way_point(self, interval=1, number_of_samples=50):
        return self.client.call("GetPathToTargetWaypoint", self._ID, interval, number_of_samples)
        
    def get_control_inputs(self):
        return self.client.call("GetVehicleControlInputs", self._ID)
        
    def get_vehicle_ground_truth(self):
        return self.client.call("GetVehicleGroundTruth", self._ID)
    
    def set_tank_model_control(self,val_right,val_left):
        return self.client.call("ApplyCustomControl",self._ID,val_right,val_left)

    def set_target_speed(self,target_speed):
        return self.client.call("SetTargetSpeed",self._ID,target_speed)

    def set_discrete_model_inputs(self,target_acc,target_steer):
        return self.client.call("SetDiscreteModelTargetAcc",self._ID,target_acc,target_steer)

    def set_tank_model_properties(self, wheel_radius, 
     vehicle_width, heading_error_degree):
        return self.client.call("SetTankModelProps", self._ID,wheel_radius,
            vehicle_width, heading_error_degree)




