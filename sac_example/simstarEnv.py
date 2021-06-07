from enum import EnumMeta
import gym 
from gym import spaces
import collections as col
import numpy as np 
import time
import sys
import os
import pickle
from statistics import mean

try:
    import simstar
except ImportError:
    print("go to PythonAPI folder where setup.py is located")
    print("python setup.py install")


class SimstarEnv(gym.Env):
    def __init__(self,track=simstar.Environments.DutchGrandPrix,
            add_opponents=False,synronized_mode=False,speed_up=1,
            host="127.0.0.1",port=8080):
        
        self.add_opponents = add_opponents # True: adds opponent vehicles; False: removes opponent vehicles
        self.number_of_opponents = 6 # agent_locations, agent_speeds, and lane_ids sizes has to be the same
        self.agent_locations = [-10, -20, -10, 0, 10, 0] # opponents' meters offset relative to ego vehicle
        self.agent_speeds = [45, 80, 55, 100, 40, 60] # opponent vehicle speeds in km/hr
        self.lane_ids = [1, 2, 3, 3, 2, 1] # make sure that the lane ids are not greater than number of lanes
        
        self.ego_lane_id = 2 # make sure that ego vehicle lane id is not greater than number of lanes
        self.ego_start_offset = 25 # ego vehicle's offset from the starting point of the road
        self.default_speed = 120 # km/hr
        self.road_width = 10


        self.track_sensor_size = 19
        self.opponent_sensor_size = 12
        
        self.time_step = 0
        self.terminal_judge_start = 100 # if ego vehicle does not have progress for 100 steps, terminate
        self.termination_limit_progress = 5 # if progress of the ego vehicle is less than 5 for 100 steps, terminate

        # the type of race track to generate 
        self.track_name = track
        
        self.synronized_mode = synronized_mode # simulator waits for update signal from client if enabled
        self.speed_up = speed_up # how faster should simulation run. up to 6x. 
        self.host = host
        self.port = port
        

        try:
            self.client = simstar.Client(host=self.host, port=self.port)
            self.client.ping()
        except (simstar.TransportError or simstar.TimeoutError):
            raise simstar.TransportError("******* Make sure a Simstar instance is open and running at port %d*******",self.port)
        
        self.client.open_env(self.track_name)
        
        print("[SimstarEnv] initializing environment")
        time.sleep(5)
        
        # get main road
        self.road = None
        all_roads = self.client.get_roads()
        if len(all_roads)>0:
            road_main = all_roads[0]
            road_id = road_main['road_id']
            self.road = simstar.RoadGenerator(self.client,road_id)

        # a list contaning all vehicles 
        self.actor_list = []

        # disable lane change for automated actors
        self.client.set_lane_change_disabled(is_disabled=True)

        #input space. 
        high = np.array([np.inf, np.inf,  1., 1.])
        low = np.array([-np.inf, -np.inf, 0., 0.])
        self.observation_space = spaces.Box(low=low, high=high)
        
        # action space: [steer, accel-brake]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.default_action = [0.0, 1.0]

    def apply_settings(self):
        self.client.set_sync_mode(self.synronized_mode,self.speed_up)

    def reset(self):
        print("[SimstarEnv] actors are destroyed")
        time.sleep(0.75)

        self.time_step = 0
        
        # delete all the actors 
        self.client.remove_actors(self.actor_list)
        self.actor_list.clear()
       
        # spawn a vehicle
        if self.track_name == simstar.Environments.DutchGrandPrix:
            vehicle_pose = simstar.PoseData(
                -603.631592, -225.756531, -3.999999,yaw=20/50)
            self.main_vehicle  = self.client.spawn_vehicle_to(vehicle_pose,
                initial_speed=0,set_speed=0,
                vehicle_type = simstar.EVehicleType.Sedan1)
        else:
            self.main_vehicle = self.client.spawn_vehicle(distance=150,lane_id=1,
                initial_speed=0,set_speed=0,
                vehicle_type = simstar.EVehicleType.Sedan1)
        print("[SimstarEnv] main v ID: ",self.main_vehicle.get_ID())

        # add all actors to the acor list
        self.actor_list.append(self.main_vehicle)

        # include other vehicles
        if self.add_opponents:

            # define other vehicles with set speeds and initial locations
            for i in range(self.number_of_opponents):
                new_agent = self.client.spawn_vehicle(
                    actor=self.main_vehicle, distance=self.agent_locations[i], lane_id=self.lane_ids[i], initial_speed=0, set_speed=self.agent_speeds[i])

                time.sleep(0.2)

                # define drive controllers for each agent vehicle
                new_agent.set_controller_type(simstar.DriveType.Auto)
                self.actor_list.append(new_agent)
            
            self.simstar_step()

            # drive all other actors in autopilot mode other than ego vehicle
            self.client.auto_pilot_agents(self.actor_list[1:])

        self.simstar_step()

        # set as display vehicle to follow from simstar
        self.client.display_vehicle(self.main_vehicle)

        # set drive type as API for ego vehicle
        self.main_vehicle.set_controller_type(simstar.DriveType.API)
        
        # attach appropriate sensors to the vehicle
        track_sensor_settings = simstar.DistanceSensorParameters(
            enable = True, draw_debug = False, add_noise = False,  position=simstar.PositionRPC(0.0, 0.0, -0.20), 
            orientation=simstar.OrientationRPC(0.0, 0.0, 0.0), minimum_distance = 0.2, maximum_distance = 200.0,
            fov = 190.0, update_frequency_in_hz = 60.0, number_of_returns=self.track_sensor_size, query_type=simstar.QueryType.Static)

        self.track_sensor = self.main_vehicle.add_sensor(simstar.ESensorType.Distance, track_sensor_settings)
        
        self.simstar_step(2)

        opponent_sensor_settings = simstar.DistanceSensorParameters(
            enable = True, draw_debug = False, add_noise = False, position=simstar.PositionRPC(2.0, 0.0, 0.4), 
            orientation=simstar.OrientationRPC(0.0, 0.0, 0.0), minimum_distance = 0.0, maximum_distance = 200.0,
            fov = 180.0, update_frequency_in_hz = 60.0, number_of_returns=self.opponent_sensor_size, query_type=simstar.QueryType.Dynamic)

        self.opponent_sensor = self.main_vehicle.add_sensor(simstar.ESensorType.Distance, opponent_sensor_settings)

        self.simstar_step(2)

        simstar_obs = self.get_simstar_obs(self.default_action)
        observation = self.make_observation(simstar_obs)
        return observation

    def calculate_reward(self, simstar_obs):
        collision = simstar_obs["damage"]
        reward = 0.0
        done = False

        trackPos =  simstar_obs['trackPos']
        angle = simstar_obs['angle']
        spx = simstar_obs['speedX']

        # progress = spx * (np.cos(angle) - np.abs(np.sin(angle)) - np.abs(trackPos))
        progress = spx * (np.cos(angle) - np.abs(np.sin(angle)))
        reward = progress

        # for debuggging purposes
        #print("angle: %2.2f,speed %2.2f, trackPos %2.2f"%(angle,sp,trackPos))
        #print("[SimstarEnv] term1 %2.2f, term2 %2.2f, term3 %2.2f, spx %2.2f, spy%2.2f"%(np.cos(angle) ,-np.abs(np.sin(angle)), -np.abs(trackPos),spx,spy ))

        # if collision, finish race
        if collision:
            print("[SimstarEnv] finish episode due to accident")
            reward = -20/3.6
            done = True
        
        # if the car has gone off road, terminate
        if abs(trackPos) > 1.0:
            print("[SimstarEnv] finish episode due to road deviation")
            reward = -20/3.6
            done = True
            
        # if the car has returned backward, end race
        if np.cos(angle) < 0.0:
            print("[SimstarEnv] finish episode due to going backwards")
            reward = -20/3.6
            done = True

        # if speed of the agent is too high, give penalty
        if spx > 120:
            reward = -(spx / 120)
        
        # if the progress of agent is small, episode terminates
        if progress < self.termination_limit_progress:
            if self.terminal_judge_start < self.time_step:
                print("[SimstarEnv] finish episode due to agent is too slow")
                reward = -20/3.6
                done = True
        else:
            self.time_step = 0

        self.progress_on_road = self.main_vehicle.get_progress_on_road()
        if self.progress_on_road > 1:
            print("[SimstarEnv] finished lap")
            done = True

        self.time_step += 1
        
        return reward, done

    def step(self, action):
        simstar_obs = self.get_simstar_obs(action)
        observation = self.make_observation(simstar_obs)
        reward, done = self.calculate_reward(simstar_obs)
        summary = {}
        return observation, reward, done, summary

    def make_observation(self, simstar_obs):
        names = ['angle', 'speedX', 'speedY', 'opponents','track','trackPos']
        Observation = col.namedtuple('Observation', names)

        return Observation( angle=np.array(simstar_obs['angle'], dtype=np.float32)/1.,
                            speedX=np.array(simstar_obs['speedX'], dtype=np.float32)/self.default_speed,
                            speedY=np.array(simstar_obs['speedY'], dtype=np.float32)/self.default_speed,
                            opponents=np.array(simstar_obs['opponents'], dtype=np.float32)/200.,
                            track=np.array(simstar_obs['track'], dtype=np.float32)/200.,
                            trackPos=np.array(simstar_obs['trackPos'], dtype=np.float32)/1.)

    def ms_to_kmh(self, ms):
        return 3.6 * ms

    def clear(self):
        self.client.remove_actors(self.actor_list)

    def end(self):
        self.clear()

    # [steer, accel, brake] input
    def action_to_simstar(self, action):
        steer = float(action[0])
        accel_brake = float(action[1])

        steer = steer * 0.5

        if accel_brake >= 0:
            throttle = accel_brake
            brake = 0.0
        else:
            brake = abs(accel_brake)
            throttle = 0.0

        self.main_vehicle.control_vehicle(steer=steer, throttle=throttle, brake=brake)
                                
    def simstar_step(self, step_num=10):
        if self.synronized_mode:
            while True:
                tick_completed = self.client.tick_given_times(step_num)
                time.sleep(0.005)
                if(tick_completed):
                    break
        else:
            time.sleep(1/60*step_num)

    def get_simstar_obs(self, action):
        self.action_to_simstar(action)

        # required to continue simulation in sync mode
        self.simstar_step()

        vehicle_state = self.main_vehicle.get_vehicle_state_self_frame()
        speed_x_kmh = abs(self.ms_to_kmh(float(vehicle_state['velocity']['X_v'])))
        speed_y_kmh = abs(self.ms_to_kmh(float(vehicle_state['velocity']['Y_v'])))
        opponents = self.opponent_sensor.get_detections()
        track = self.track_sensor.get_detections()
        road_deviation = self.main_vehicle.get_road_deviation_info()

        retry_counter = 0
        while len(track) < self.track_sensor_size or len(opponents) < self.opponent_sensor_size:
            self.simstar_step()
            time.sleep(0.1)
            opponents = self.opponent_sensor.get_detections()
            track = self.track_sensor.get_detections()
            retry_counter += 1
            if retry_counter > 1000: raise RuntimeError("Track Sensor shape error. Exited")
        
        speed_x_kmh = np.sqrt((speed_x_kmh*speed_x_kmh) + (speed_y_kmh*speed_y_kmh))
        speed_y_kmh = 0.0
        
        # deviation from road in radians
        angle = float(road_deviation['yaw_dev'])
        
        # deviation from road center in meters
        trackPos = float(road_deviation['lat_dev']) / self.road_width

        # if collision occurs, True. else False
        damage = bool( self.main_vehicle.check_for_collision())

        simstar_obs = {
            'angle': angle,
            'speedX': speed_x_kmh,
            'speedY':speed_y_kmh,
            'opponents':opponents ,
            'track': track,                
            'trackPos': trackPos,
            'damage':damage
            }
        return simstar_obs

    def __del__(self):
        # reset sync mod so that user can interact with simstar
        if(self.synronized_mode):
            self.client.set_sync_mode(False)

if __name__ =='__main__':
    env = SimstarEnv(simstar.Environments.CircularRoad,add_opponents=True)
    env.reset()
    env.main_vehicle.set_controller_type(simstar.DriveType.Keyboard)
    
    while 1:
        pr = env.main_vehicle.get_progress_on_road()
        print(pr)