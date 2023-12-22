import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from datetime import date
from tqdm import tqdm
import tensorflow as tf
#import keras.backend.tensorflow_backend as backend
from keras.models import load_model

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
sys.path.append("/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg")
import carla
import carla_config as settings
import re
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)


class CarEnv:
    im_width = settings.IM_WIDTH_VISUALIZATION  # width of the image that we want have
    im_height = settings.IM_HEIGHT_VISUALIZATION  # height of the image that we want have
    front_camera = None
    angle_rw = 0
    trackpos_rw = 0
    cmd_vel = 0
    summary = {'Target': 0, 'Steps': 0}
    distance_acum = []

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        # self.map = self.world.get_map()
        # self.dao = GlobalRoutePlannerDAO(self.map, 2.0)
        # self.grp = GlobalRoutePlanner(self.map)
        self.prev_d2goal = 10000
        self.Target = 0
        self.numero_tramo = 0 # numero_tramo = section_number
        self.error_lateral = []
        self.position_array = []
        self.prev_next = 0
        self.waypoints_txt = []
        # self.model_waypoints = load_model('/home/robesafe/AAOscar/TRAIN_ROAD/PilotNet_100BOVRGB_good.model')
        #self.model_waypoints = load_model(settings.PRE_CNN_PATH)

        #############NUEVO #### new
        if settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[1]:
            self.pos_a = carla.Transform(carla.Location(x=328.712982, y=195.114639, z=1.000000),
                                         carla.Rotation(pitch=0.000000, yaw=180.004654, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=88.415741, y=300.859680, z=1.000000),
                                         carla.Rotation(pitch=0.000000, yaw=89.991280, roll=0.000000))

        elif settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[2]:
            self.pos_a = carla.Transform(carla.Location(x=208.669876, y=195.149597, z=1.000000),
                                         carla.Rotation(pitch=360.000000, yaw=180.004654, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=88.415741, y=300.859680, z=1.000000),
                                         carla.Rotation(pitch=0.000000, yaw=89.991280, roll=0.000000))

        elif settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[3]:
            self.pos_a = carla.Transform(carla.Location(x=150.669876, y=195.149597, z=1.000000),
                                         carla.Rotation(pitch=360.000000, yaw=180.004654, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=92.385292, y=100.597343, z=1.000000),
                                         carla.Rotation(pitch=360.000000, yaw=269.991272, roll=0.000000))

        elif settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[4]:
            # self.pos_a = carla.Transform(carla.Location(x=196.748154, y=55.487041, z=1.000000),
            #                         carla.Rotation(pitch=360.000000, yaw=179.993011, roll=0.000000))
            self.pos_a = carla.Transform(carla.Location(x=173.748154, y=55.487041, z=1.000000),
                                         carla.Rotation(pitch=360.000000, yaw=179.993011, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=109.849731, y=-2.049278, z=1.000000),
                                         carla.Rotation(pitch=0.000000, yaw=-179.993881, roll=0.000000))



        elif settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[5]:
            self.pos_a = carla.Transform(carla.Location(x=300.351135, y=59.474419, z=1.000000),
                                         carla.Rotation(pitch=0.000000, yaw=-0.006982, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=379.485901, y=2.017289, z=1.000000),
                                         carla.Rotation(pitch=360.000000, yaw=0.030457, roll=0.000000))

        else:
            self.pos_a = 0
            self.pos_b = 0

        self.ind = 1

        src = np.float32([[0, settings.IM_HEIGHT_VISUALIZATION], [1200, settings.IM_HEIGHT_VISUALIZATION], [0, 0],
                          [settings.IM_WIDTH_VISUALIZATION, 0]])
        dst = np.float32([[569, settings.IM_HEIGHT_VISUALIZATION], [711, settings.IM_HEIGHT_VISUALIZATION], [0, 0],
                          [settings.IM_WIDTH_VISUALIZATION, 0]])
        self.M = cv2.getPerspectiveTransform(src, dst)
    # return im, state_train
    # return im, simple_state
    # return self.front_camera, self.state_train
    @property
    def reset(self):
        global acum
        global x_prev
        global y_prev
        acum = 0
        self.collision_hist = []
        self.actor_list = []
        self.crossline_hist = []
        self.coeficientes = np.zeros((51 - 1, 8))
        self.pos_array_wp = 0
        self.waypoints_current_plan = []

        #############################NUEVO
        self.d2goal = 1
        self.map = self.world.get_map()
        #self.dao = GlobalRoutePlannerDAO(self.map, 1.0)
        self.grp = GlobalRoutePlanner(self.map,1)
        #self.grp.setup()
        #############################

        # aux_position = random.sample(self.positions, 1)
        # self.transform = carla.Transform(carla.Location(x=26.638832, y=-20.751266, z=4.000000),
        # carla.Rotation(pitch=-0.171233, yaw=-44.747993, roll=-0.000488))

        # self.transform = carla.Transform(
        # carla.Location(x=aux_position[0][0], y=aux_position[0][1], z=aux_position[0][2]),
        # carla.Rotation(pitch=aux_position[0][3], yaw=aux_position[0][4], roll=aux_position[0][5]))
        # self.ind = 1
        # Train_mode_options[6] == "ALTERNATIVE"
        if settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[6]:
            if self.ind == 0:
                # self.pos_a = carla.Transform(carla.Location(x=196.748154, y=55.487041, z=1.000000),
                #                              carla.Rotation(pitch=360.000000, yaw=179.993011, roll=0.000000))
                # self.pos_b = carla.Transform(carla.Location(x=109.849731, y=-2.049278, z=1.000000),
                #                              carla.Rotation(pitch=0.000000, yaw=-179.993881, roll=0.000000))
                self.pos_a = carla.Transform(carla.Location(x=328.712982, y=195.114639, z=1.000000),
                                             carla.Rotation(pitch=0.000000, yaw=180.004654, roll=0.000000))
                self.pos_b = carla.Transform(carla.Location(x=88.415741, y=300.859680, z=1.000000),
                                             carla.Rotation(pitch=0.000000, yaw=89.991280, roll=0.000000))
                self.ind = 1

            elif self.ind == 1:
                self.pos_a = carla.Transform(carla.Location(x=173.748154, y=55.487041, z=1.000000),
                                             carla.Rotation(pitch=360.000000, yaw=179.993011, roll=0.000000))
                self.pos_b = carla.Transform(carla.Location(x=109.849731, y=-2.049278, z=1.000000),
                                             carla.Rotation(pitch=0.000000, yaw=-179.993881, roll=0.000000))
                self.ind = 2

            elif self.ind == 2:
                self.pos_a = carla.Transform(carla.Location(x=158.0, y=15.487041, z=1.000000),
                                             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.0))
                self.pos_b = carla.Transform(carla.Location(x=109.849731, y=-2.049278, z=1.000000),
                                             carla.Rotation(pitch=0.000000, yaw=-179.993881, roll=0.000000))
                self.ind = 1

        #############NUEVO
        #Train_mode_options[6] == "RANDOM"
        if settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[0]:
            spawn_points = self.map.get_spawn_points()
            self.waypoints_current_plan = []
            # while self.d2goal > 200 or self.d2goal < 180:
            while self.d2goal < 400:
                self.pos_a = random.choice(spawn_points)
                self.pos_b = random.choice(spawn_points)

                a = self.pos_a.location
                b = self.pos_b.location
                self.current_plan = self.grp.trace_route(a, b)
                self.d2goal = self.total_distance(self.current_plan)

            self.current_plan = self.current_plan[:200]
            self.d2goal = self.total_distance(self.current_plan)

            self.transform = self.pos_a

        else:
            self.current_plan = self.grp.trace_route(self.pos_a.location, self.pos_b.location)
            self.current_plan = self.current_plan[:200]
            self.d2goal = self.total_distance(self.current_plan)

            self.transform = self.pos_a

        for i in range(len(self.current_plan)):
            w1 = self.current_plan[i][0]
            self.waypoints_current_plan.append(
                [w1.transform.location.x, w1.transform.location.y, w1.transform.location.z,
                 w1.transform.rotation.pitch, w1.transform.rotation.yaw, w1.transform.rotation.roll])
        self.waypoints_current_plan.append([0, 0, 0, 0, 0, 0])
        # the last point from the for
        self.Target = w1.transform.location
        ##################
        # if settings.DRAW_TRAJECTORY == 1:
        #     self.draw_path(self.world, self.current_plan)

        self.draw_path(self.world, self.current_plan[0:100], 15)

        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        self.actor_list.append(self.vehicle)
        # WORKING_MODE_OPTIONS[5] == "CNN_SEMANTIC" attach a rgb or semantic segmentation cam considering the working
        # mode options
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[5]:
            self.rgb_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        else:
            self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')

        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")

        # WORKING_MODE_OPTIONS[0] == "WAYPOINTS_CARLA"
        # WORKING_MODE_OPTIONS[8] == "TP_ANG"
        # WORKING_MODE_OPTIONS[9] == ""PRE_TRAINED_CNN"
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
                settings.WORKING_MODE_OPTIONS[8] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            # transform = carla.Transform(carla.Location(x=25, z=55), carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
            # transform = carla.Transform(carla.Location(x=3.5, z=2.5), carla.Rotation(pitch=-40.0, yaw=0.0, roll=0.0)) #ORiginal para640x480 BUENA PARA WP_CARLA
            # transform = carla.Transform(carla.Location(x=5, z=20), carla.Rotation(pitch=-75.0, yaw=0.0, roll=0.0)) ## BOV
            # transform = carla.Transform(carla.Location(x=0, z=3.5), carla.Rotation(pitch=-25.0, yaw=0.0, roll=0.0)) ## NO BOV (PilotNet_100RGB_good.model (50x50))
            # transform = carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-20.0, yaw=0.0, roll=0.0)) ## NO BOV (PilotNet_100RGB_good.model (50x50))
            transform = carla.Transform(carla.Location(x=settings.CAM_X, z=settings.CAM_Z),
                                        carla.Rotation(pitch=settings.CAM_PITCH, yaw=settings.CAM_YAW,
                                                       roll=settings.CAM_ROLL))  # ORiginal para640x480

        else:
            # transform = carla.Transform(carla.Location(x=3.5, z=2.8), carla.Rotation(pitch=-45.0, yaw=180.0, roll=0.0))
            # transform = carla.Transform(carla.Location(x=5, z=20), carla.Rotation(pitch=-75.0, yaw=180.0, roll=0.0))
            transform = carla.Transform(carla.Location(x=3.5, z=2.5),
                                        carla.Rotation(pitch=-40.0, yaw=0.0, roll=0.0))  # ORiginal para640x480

        # Attach the cameras on the vehicle
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform,
                                             attach_to=self.vehicle)  # , attachment_type=carla.AttachmentType.SpringArm)
        self.actor_list.append(self.sensor)
        # called its time a new image is generated by the sensor to save the data to the disk
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        # This sensor, when attached to an actor, it registers an event each time the actor collisions against
        # something in the world.
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        # it registers an event each time the actor crosses a lane marking.
        x_linesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.x_linesensor = self.world.spawn_actor(x_linesensor, transform, attach_to=self.vehicle)
        # reports its current gnss position.
        gnss_sensor = self.blueprint_library.find("sensor.other.gnss")
        self.gnss_sensor = self.world.spawn_actor(gnss_sensor, transform, attach_to=self.vehicle)

        self.colsensor.listen(lambda event: self.collision_data(event))
        self.x_linesensor.listen(lambda event2: self.crossline_data(event2))
        self.gnss_sensor.listen(lambda event3: self.gnss_data(event3))

        while self.front_camera is None:
            time.sleep(0.01)
        # returns the time as a floating point number expressed in seconds since the epoch, in UTC
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        location_reset = self.vehicle.get_transform()
        x_prev = location_reset.location.x
        y_prev = location_reset.location.y

        # self.state_train = self.Calcular_estado(self.front_camera)
        # WORKING_MODE_OPTIONS[1] == "WAYPOINTS_IMAGE"
        # Calcular_estado, transform2local returns the state and the exit flag
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
            # estado = situation
            self.state_train, _ = self.Calcular_estado(self.front_camera)
            return self.front_camera, self.state_train
        else:
            im = cv2.resize(self.front_camera, (settings.IM_WIDTH_CNN, settings.IM_HEIGHT_CNN))
            state_train, _ = self.transform2local(im)  # Reset flag and start iterating until episode ends

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
                simple_state = np.array([self.trackpos_rw, self.angle_rw])
                return im, simple_state
            else:
                return im, state_train

    # sum the individual distances between its wp that the agent cross and return the total_distance
    def total_distance(self, current_plan):
        sum = 0
        for i in range(len(current_plan) - 1):
            sum = sum + self.distance_wp(current_plan[i + 1][0], current_plan[i][0])
        return sum

    # calculate the Euclidean distance between two carla waypoints
    def distance_wp(self, target, current):
        dx = target.transform.location.x - current.transform.location.x
        dy = target.transform.location.y - current.transform.location.y
        return math.sqrt(dx * dx + dy * dy)

    # calculate the Euclidean distance between two carla points
    def distance_target(self, target, current):
        dx = target.x - current.x
        dy = target.y - current.y
        return math.sqrt(dx * dx + dy * dy)

    # draw green lines between individual waypoints and draw the total path
    def draw_path(self, world, current_plan, life_t):
        for i in range(len(current_plan) - 1):
            w1 = current_plan[i][0]
            w2 = current_plan[i + 1][0]

            self.world.debug.draw_line(w1.transform.location, w2.transform.location, thickness=0.1,
                        color=red, life_time=life_t)

            # world.debug.draw_point(w1.transform.location, 0.1, red, life_t)

        # self.draw_waypoint_info(world, current_plan[-1][0])

    # return the location of a waypoint and draw it on the map
    def draw_waypoint_info(self, world, w, lt=(settings.SECONDS_PER_EPISODE + 5.0)):
        w_loc = w.transform.location
        world.debug.draw_point(w_loc, 0.5, red, lt)

    # data from line 260(event2). Data if the vehicle cross a lane
    def crossline_data(self, event2):  # More soft condition could be possible
        # lane_types = set(x.type for x in event2.crossed_lane_markings)
        # text = [str(x).replace("['", "") for x in lane_types]
        # if (str(text).replace("['", "").replace("']", "") == 'NONE') or\
        #         (str(text).replace("['", "").replace("']", "") == 'Broken'):
        self.crossline_hist.append(1)
    # data from line 263(event3). return the latitude and longitude from the data of the event 3
    def gnss_data(self, event3):
        global latitude
        global longitude

        latitude = event3.latitude
        longitude = event3.longitude
    # data from the line 256(event).
    def collision_data(self, event):
        self.collision_hist.append(1)

    def process_img(self, image):
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
                settings.WORKING_MODE_OPTIONS[1] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
            i = np.array(image.raw_data)
            i2 = i.reshape((self.im_height, self.im_width, 4))
            i3 = i2[:, :, :3]
            if settings.BEV_PRE_CNN == 1:
                self.front_camera = cv2.warpPerspective(i3, self.M,
                                                        (settings.IM_WIDTH_VISUALIZATION,
                                                         settings.IM_HEIGHT_VISUALIZATION))
            else:
                self.front_camera = i3

        else:
            if settings.THRESHOLD == 0:
                if settings.IM_TYPE == 1:
                    image.convert(carla.ColorConverter.CityScapesPalette)
                i = np.array(image.raw_data)
                i2 = i.reshape((self.im_height, self.im_width, 4))
                i3 = i2[:, :, :3]
                gray = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)

                if settings.IM_LAYERS == 1:
                    self.front_camera = gray
                elif settings.IM_LAYERS == 3:
                    if settings.BEV_PRE_CNN == 1:
                        self.front_camera = cv2.warpPerspective(i3, self.M,
                                                                (settings.IM_WIDTH_VISUALIZATION,
                                                                 settings.IM_HEIGHT_VISUALIZATION))
                    else:
                        self.front_camera = i3
            else:
                i = np.array(image.raw_data)
                i2 = i.reshape((self.im_height, self.im_width, 4))
                i3 = i2[:, :, :3]
                kernel = np.ones((5, 5), np.uint8)
                # kernel = np.ones((6, 6), np.uint8)
                ang_deg = 0.0
                mask = cv2.inRange(i3, (0, 200, 0), (10, 256, 10))
                gray = cv2.dilate(mask, kernel, iterations=2)
                gray = cv2.erode(gray, kernel, iterations=2)
                self.front_camera = gray

    # return [im, next], reward, done, None
    # return [self.front_camera, state], reward, done, None
    # return [im, simple_state], reward, done, None
    def step(self, action):
        global x_prev
        global y_prev
        global acum
        global acum_prev
        global d_i_prev
        # Action is applied like steerin while throttle is cte
        # if settings.ACTIONS_NAMES[action] != settings.ACTIONS_NAMES[settings.N_actions-1]:
        # action_control a dict with values for brake,steer and brake
        self.vehicle.apply_control(carla.VehicleControl(throttle=settings.ACTION_CONTROL[action][0],
                                                        brake=settings.ACTION_CONTROL[action][1],
                                                        steer=settings.ACTION_CONTROL[action][2]))
        # if action == 0:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.50, steer=-1 * settings.STEER_AMT))
        # elif action == 1:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.55, steer=0))
        # elif action == 2:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.50, steer=1 * settings.STEER_AMT))

        # v = self.vehicle.get_velocity()
        # kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        # if kmh > 120:
        #     kmh = 120

        # print("acumulado: ", acum)
        # take the current location-rotation of the vehicle
        location_rv = self.vehicle.get_transform()
        # print(self.vehicle.get_location())

        # Se tiene un waypoint de carla
        # location = self.vehicle.get_location()
        # euclidean distance
        d_i = math.sqrt((x_prev - location_rv.location.x) ** 2 + (y_prev - location_rv.location.y) ** 2)
        # sum the d_i that the agent pass by
        acum += d_i
        # make the current loc and rot the previous
        x_prev = location_rv.location.x
        y_prev = location_rv.location.y
        # save the waypoints that the agents pass by
        self.position_array.append(
            [x_prev, y_prev, location_rv.location.z, location_rv.rotation.pitch, location_rv.rotation.yaw,
             location_rv.rotation.roll])
        # print(reward)
        x_prev = location_rv.location.x
        y_prev = location_rv.location.y
        d_i_prev = d_i

        reward, done, d2target = self.get_reward()

        if settings.SHOW_CAM == 1:
            # used to create a window with a suitable name and size to display images and videos on the screen
            cv2.namedWindow('Real', cv2.WINDOW_AUTOSIZE)
            # method is used to display an image in a window
            cv2.imshow('Real', self.front_camera)
            # allows users to display a window for given milliseconds or until any key is pressed
            cv2.waitKey(1)

        # SALIDA UTILIZADA PARA EL PROGRAMA DE LOS WAYPOINTS OBTENIDOS POR TRATAMIENTO DE IMAGEN
        # OUTPUT USED FOR THE PROGRAM OF THE WAYPOINTS OBTAINED BY IMAGE PROCESSING
        # WORKING_MODE_OPTIONS[1] = "WAYPOINTS_IMAGE"
        # means that crush os something bad happens bcs exit flag =1
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
            # im = cv2.resize(self.front_camera, (settings.IM_WIDTH_CNN, settings.IM_HEIGHT_CNN))

            state, exit_flag = self.Calcular_estado(self.front_camera)
            # if vehicle crush
            if exit_flag == 1:
                print('Waypoints, distance to target have been lost: ', d2target)
                done = True
                reward = -200
            # if vehicle crush
            if done == True:
                # we append the total distance that pass by and save the total distance
                self.distance_acum.append(acum)

            return [self.front_camera, state], reward, done, None

        # SALIDA UTILIZADA EL RESTO DE PROGRAMAS
        # OUTPUT USED THE REST OF PROGRAMS
        else:
            im = cv2.resize(self.front_camera, (settings.IM_WIDTH_CNN, settings.IM_HEIGHT_CNN))
            next, exit_flag = self.transform2local(im)
            # SI HA DADO UN BANDAZO Y NO SE VE NINGUN WAYPOINT DELANTE SE SALE
            # IF YOU HAVE TURNED AND YOU CANNOT SEE ANY WAYPOINT AHEAD, YOU WILL LEAVE

            # Comprobar si en la imgaen BW segmentada se sale de la carretera
            # Check if the segmented BW image goes off the road
            # means that crush os something bad happens bcs exit flag =1
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[2] or settings.WORKING_MODE == \
                    settings.WORKING_MODE_OPTIONS[7]:
                if np.count_nonzero(
                        # If there are not a thousand white dots we say that we have left
                        self.front_camera) < 1000:  # si no hay mil puntos blancos decimos que nos hemos salido
                    exit_flag = 1

            # Comprobar si en la imagen RGB con carril se pierde el camino
            # Check if the path is lost in the RGB image with rail
            # print(np.sum(self.front_camera[:, :, 1] == 234), np.sum(self.front_camera[:, :, 1] == 220))
            # means that crush os something bad happens bcs exit flag =1
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[4] or settings.WORKING_MODE == \
                    settings.WORKING_MODE_OPTIONS[8] \
                    or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0]:
                if np.sum(self.front_camera[:, :, 1] == 234) <= 50 and np.sum(self.front_camera[:, :,
                                                                              1] == 220) <= 260:  # si no hay mil puntos blancos decimos que nos hemos salido
                    exit_flag = 1

            # print(self.front_camera[:, :, 1])
            if exit_flag == 1:
                print('Waypoints have been lost, distance to the target: ', d2target)
                done = True
                reward = -200

            im = cv2.resize(self.front_camera, (settings.IM_WIDTH_CNN, settings.IM_HEIGHT_CNN))
            if settings.SHOW_CAM_RESIZE == 1:
                cv2.namedWindow('Resize', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Resize', im)
                cv2.waitKey(1)

            if done == True:
                # we append the total distance that pass by
                self.distance_acum.append(acum)

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
                simple_state = np.array([self.trackpos_rw, self.angle_rw])
                return [im, simple_state], reward, done, None
            else:
                return [im, next], reward, done, None

    # return reward, done, d2target
    def get_reward(self):
        # take the velocity of the vehicle
        v = self.vehicle.get_velocity()
        # from m/s to km/h
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        # reduce speed
        if kmh > 120:
            kmh = 120

        location = self.vehicle.get_location()
        # cos(angle_rw)-|sin(angle_rw)|-|trackpos_rw|
        # angle_rw and trackpos_rw calculated on the def transform2local or Calcular_estado
        progress = np.cos(self.angle_rw) - abs(np.sin(self.angle_rw)) - abs(self.trackpos_rw)
        # CONDICIÓN DE SALIDA DEL PROGRAMA.
        # PROGRAM EXIT CONDITION
        # salida = exit
        salida = 0
        # the distance between the cur loc and the next loc or the final loc
        d2target = self.distance_target(self.Target, location)
        # CONDICIÓN DE SALIDA SI HAY COLISIÓN
        # EXIT CONDITION IF THERE IS A COLLISION
        if len(self.collision_hist) != 0:  # or (len(self.crossline_hist) != 0)
            done = True
            # salida = 1 means that u must do it again the simulation
            salida = 1
            # very high negative reward that means something goes very, very bad
            reward = -200
            print('There has been a collision, distance to target: ', d2target)
            # goes to the next step/run the simulation again
            self.summary['Steps'] += 1

        # CONDICIÓN DE SALIDA SI HAY SALIDA DE CARRIL
        # if len(self.crossline_hist) != 0:
        #     done = True
        #     salida = 1
        #     reward = -200
        #     print('Ha habido una salida de carril, distancia al objetivo: ', d2target)
        #     self.summary['Steps'] += 1
        # SI NO HAY CONDICION DE SLAIDA DEL PROGRAMA
        # IF THERE IS NO PROGRAM RELEASE CONDITION
        if salida == 0:

            # SE LE DA LA RECOMPENSA EN FUNCION DE COMO VAYA EN LA CARRETERA
            # THE REWARD IS GIVEN TO YOU BASED ON HOW YOU DO ON THE ROAD
            # the reward given based on the progress u have done
            # modo_recompensa = reward_mode
            # # angle_rw and trackpos_rw calculated on the def transform2local or Calcular_estado
            if settings.modo_recompensa == 0:
                if kmh < 10:
                    done = False
                    reward = -1
                else:
                    done = False
                    reward = 1
            elif settings.modo_recompensa == 1:
                reward = progress
                done = False
            else:
                reward = (kmh) * progress
                done = False

            # SI HA LLEGADO AL OBJETIVO SE CAMBIA LA RECOMPENSA Y SE SALE
            # IF YOU HAVE REACHED THE GOAL, THE REWARD IS CHANGED AND YOU EXIT
            if self.distance_target(self.Target, location) < 15:
                done = True
                reward = 100
                self.summary['Steps'] += 1
                self.summary['Target'] += 1
                print('The goal has been reached')

            # SI SE HA FINALIZADO EL TEMPORIZADOR SE CAMBIA LA RECOMPENSA Y SE SALE
            # IF THE TIMER HAS ENDED, THE REWARD IS CHANGED AND THEY EXIT
            if self.episode_start + settings.SECONDS_PER_EPISODE < time.time():
                print('End of timer, distance to target: ', d2target)
                done = True
                self.summary['Steps'] += 1
                if acum <= 50:
                    reward = -200
                elif (acum > 50) and (acum < 160):
                    reward = -100
                else:
                    reward = 100
        # normalizo la velocidad
        # normalize the speed
        self.cmd_vel = kmh / 120
        ###### When an important fact happened the done = true and steps += 1
        return reward, done, d2target

    # Calcular_estado = calculate_state
    # return state, exit_flag
    # detect road features (road edges,road lanes, angles) and determining whether the vehicle might have left the road.
    def Calcular_estado(self, img2):
        global center_old
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(img2, (0, 200, 0), (10, 256, 10))
        gray = cv2.dilate(mask, kernel, iterations=2)
        gray = cv2.erode(gray, kernel, iterations=2)
        exit_flag = 0
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        height = gray.shape[0]
        width = gray.shape[1]
        waypoint = np.zeros((15,))
        waypoint_edges = np.zeros((15, 2))
        # the size of the input state is 16
        state = np.zeros((settings.state_dim,))

        # CALCULO DEL PUNTO DE FUGA
        # CALCULATION OF THE VANISHING POINT
        # determines where is the lane of the road  via waypoints and adjust these waypoints
        # final draw these waypoints to see where is the road
        for i in range(0, 15):
            dato_y = int(height - 1 - 25 * i)

            for j in range(0, width):
                if gray[dato_y, j] == 255:
                    waypoint_edges[i][
                        0] = j  # me quedo con la coordenada x empezando por la izqueirda de la imagen, la y viene dada por el indice
                    break
            for j in range(0, width):
                if gray[dato_y, width - 1 - j] == 255:
                    waypoint_edges[i][
                        1] = width - 1 - j  # me quedo con la coordenada x empezando por la izqueirda de la imagen, la y viene dada por el indice
                    break
            waypoint[i] = int((waypoint_edges[i][0] + waypoint_edges[i][1]) / 2)
            if i < 6:
                if (waypoint_edges[i][0] == 0) and (waypoint_edges[i][1] < (width - 1)):
                    waypoint[i] = waypoint_edges[i][1] - (280 - 20 * i)
                    if waypoint[i] < 0:
                        waypoint[i] = 0
                elif (waypoint_edges[i][0] > 0) and (waypoint_edges[i][1] >= (width - 1)):
                    waypoint[i] = waypoint_edges[i][0] + (280 - 20 * i)
                    if waypoint[i] > (width - 1):
                        waypoint[i] = width - 1

            if i == 1:
                waypoint[0] = waypoint[1]
            # waypoint[i] = int((waypoint_edges[i][0] + waypoint_edges[i][1]) / 2)

        # PINTAR LOS PUNTOS DE LA CARRETERA
        # PAINT THE POINTS ON THE ROAD
        for i in range(0, 15):
            dato_y = int(height - 1 - 25 * i)
            waypointcenter2 = (int(waypoint[i]), int(dato_y))
            cv2.circle(gray, waypointcenter2, 2, (0, 0, 0), 2)
            waypoint[i] = (waypoint[i] - width / 2) / (width / 2)
            if i > 1 and waypoint[i] == -1:
                if waypoint[i - 1] > 0.3:
                    waypoint[i] = 1
                elif waypoint[i - 1] < -0.3:
                    waypoint[i] = -1
                else:
                    waypoint[i] = 0

        # CALCULAR EL ÁNGULO DE LA CARRETERA
        # Computes the angle of the road, based on the waypoints
        x_diff = waypoint[5] - waypoint[7]
        y_diff = (7 * 25 - 5 * 25) / (width / 2)
        # we calculate the const angle_rw
        self.angle_rw = np.arctan2(x_diff, y_diff)
        # state 0--->14 waypoints
        # state 15 angle_rw
        # state 16 cmd_vel (speed)
        state[0:(settings.state_dim - 2)] = waypoint
        state[settings.state_dim - 2] = self.angle_rw / math.pi
        state[settings.state_dim - 1] = self.cmd_vel
        # we calculate the const trackpos_rw provide information about how centered the vehicle is to the detected
        # lane or how many metres is from the road edges.
        self.trackpos_rw = waypoint[0]

        if settings.SHOW_WAYPOINTS == 1:
            cv2.namedWindow('Punto de fuga', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Punto de fuga', gray)
            cv2.waitKey(1)
        # si no hay mil puntos blancos decimos que nos hemos salido
        # If there are not a thousand white dots means that we have detected that we are out of the road
        if np.count_nonzero(gray) < 1000:
            exit_flag = 1

        return state, exit_flag
    # return state, exit_flag
    # transforming waypoints from a global coordinate system to a local one
    # and predict the next state via next15 or ANN
    def transform2local(self, im):
        state = np.zeros((settings.dimension_vector_estado,))
        actual_pos = self.vehicle.get_transform()
        yaw_c = actual_pos.rotation.yaw * math.pi / 180 - math.pi / 2
        exit_flag = 0
        # print('yaw_C: ',yaw_c)
        Xc = actual_pos.location.x
        Yc = actual_pos.location.y
        Zc = actual_pos.location.z
        # print('Pc :', Xc, ' ', Yc, ' ', Zc)
        self.waypoints_current_plan[-1] = [actual_pos.location.x, actual_pos.location.y, actual_pos.location.z,
                                           actual_pos.rotation.pitch, actual_pos.rotation.yaw,
                                           actual_pos.rotation.roll]

        aux_waypoints = np.array(self.waypoints_current_plan)
        self.waypoints_txt = aux_waypoints

        aux_waypoints = aux_waypoints[0:-1, 0:4]
        aux_waypoints[:, 3] = 1

        # aux_waypoints[:, 0] = -aux_waypoints[:, 0]

        M = np.array(([np.cos(yaw_c), -np.sin(yaw_c), 0, Xc],
                      [np.sin(yaw_c), np.cos(yaw_c), 0, Yc],
                      [0, 0, 1, Zc],
                      [0, 0, 0, 1]))
        # print('WP1: ', aux_waypoints[0, :])
        # print('WP-1: ', aux_waypoints[-1, :])
        # inverse the matrix
        M_inv = np.linalg.inv(M)
        # convert to local coordinates
        P_locales = np.zeros((len(aux_waypoints), 4))
        # plt.figure(1)
        for i in range(len(aux_waypoints)):
            P_locales[i] = np.dot(M_inv, aux_waypoints[i, :])
        P_locales[:, 0] = -P_locales[:, 0]
        P_locales_aux = P_locales[self.pos_array_wp:(self.pos_array_wp + 30)]
        # Pintar el número de waypoints que se han pasado
        # Paint the number of waypoints that have been passed
        wp_out = np.where(P_locales_aux[:, 1] < 0)
        n_wp_out = len(wp_out[0])
        # the next wp determined if the vehicle hasn't pass from this wp
        nextWP = P_locales_aux[n_wp_out:(n_wp_out + 15)]
        self.pos_array_wp += n_wp_out

        if n_wp_out != 0:
            self.draw_path(self.world, self.current_plan[self.pos_array_wp:(self.pos_array_wp + 100)], 5)

        next15_aux = nextWP[:, 0]
        next15_aux_y = nextWP[:, 1]

        nextt15_aux = nextWP[:, 0:2]
        # SE COMPRUEBA EL TAMAÑO DEL VECTOR DE WAYPOINTS, SI ES MENOR DE 15 SE ALARGA EL ÚLTIMO VALOR HASTA EL FINAL.
        # THE SIZE OF THE WAYPOINT VECTOR IS CHECKED, IF IT IS LESS THAN 15 THE LAST VALUE IS EXTENDED UNTIL THE END.
        next15 = np.zeros((15, 2))
        tam_wp = len(nextt15_aux)
        if tam_wp < 15:
            if tam_wp == 0:
                exit_flag = 1
                next15 = self.prev_next
            else:
                exit_flag = 0
                next15[0:tam_wp] = nextt15_aux
                for k in range(15 - tam_wp):
                    next15[-1 - k] = nextt15_aux[tam_wp - 1]
        else:
            exit_flag = 0
            next15 = nextt15_aux

        # # para corregir las salidas inesperadas
        # if nextt15_aux[0][1] < -2.5:
        #     print("Posible salida de episodio menor -2.5 ", nextt15_aux[0][1])
        #     exit_flag = 1
        # if nextt15_aux[0][1] > 12:
        #     print("Posible salida de episodio mayor 12 ", nextt15_aux[0][1])
        #     exit_flag = 1

        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
                settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            # if settings.SHOW_WAYPOINTS == 1:
            # DIBUJAR LOS PUNTOS EN OPENCV
            img_negra = np.zeros((512, 512, 3), np.uint8)
            # print(next15/100)
            for i in range(len(next15)):
                pto = (int(next15[i][0] * 20 + 512 / 2), int(512 - next15[i][1] * 30))
                cv2.circle(img_negra, pto, 3, (255, 0, 0), 2)

            if settings.SHOW_WAYPOINTS == 1:
                cv2.namedWindow('Waypoints', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Waypoints', img_negra)
                cv2.waitKey(1)

        # gray2 = cv2.cvtColor(img_negra, cv2.COLOR_BGR2GRAY)
        # if np.max(cv2.cvtColor(img_negra, cv2.COLOR_BGR2GRAY)) == 0:
        #     exit_flag = 1

        self.prev_next = next15


        # x_diff = next15[7][1] - next15[4][1]
        # y_diff = -(next15[7][0] - next15[4][0])

        x_diff = next15[5][1] - next15[2][1]
        y_diff = -(next15[5][0] - next15[2][0])

        self.angle_rw = np.arctan2(y_diff, x_diff)
        # print('Angulo de muestra:', 180-(prueba_angle_rw*180/np.pi))
        self.trackpos_rw = next15[0][0]

        # print(exit_flag)
        # calculate the new_state_state the predicted one
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            waypoints_predicted = self.model_waypoints.predict(
                np.array(im).reshape(-1, settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, 3) / 255, verbose=0)
            waypoints_predicted = waypoints_predicted.reshape(15, 2)

            # waypoints[:, 0] = -waypoints[:, 0]
            if settings.SHOW_WAYPOINTS == 1:
                # DIBUJAR LOS PUNTOS EN OPENCV
                # DRAW THE POINTS IN OPENCV

                for i in range(len(waypoints_predicted)):
                    pto = (int(waypoints_predicted[i][0] * 20 + 512 / 2), int(512 - waypoints_predicted[i][1] * 30))
                    cv2.circle(img_negra, pto, 3, (0, 0, 255), 2)

                cv2.namedWindow('Waypoints', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Waypoints', img_negra)
                cv2.waitKey(1)

            if settings.WAYPOINTS == 'XY':
                state[0:(settings.dimension_vector_estado - 1)] = waypoints_predicted.flatten()
                state[settings.dimension_vector_estado - 1] = self.angle_rw / np.pi
                return state, exit_flag

            elif settings.WAYPOINTS == 'X':
                # print('Waypoints de CNN')
                state[0:(settings.dimension_vector_estado - 1)] = waypoints_predicted[:, 0] / 20
                state[settings.dimension_vector_estado - 1] = self.angle_rw / math.pi
                return state, exit_flag

        # Se devuelve el valor normalizado entre 100 metros
        # The normalized value is returned between 100 meters
        # calculate the next state by the vector next15(its now predicted wp but from grp)
        if settings.WAYPOINTS == 'XY':
            state[0:(settings.dimension_vector_estado - 1)] = next15.flatten()
            state[settings.dimension_vector_estado - 1] = self.angle_rw
            return state, exit_flag

        elif settings.WAYPOINTS == 'X':
            # print('Waypoints de carla')
            state[0:(settings.dimension_vector_estado - 1)] = next15[:, 0] / 20
            state[settings.dimension_vector_estado - 1] = self.angle_rw / math.pi
            # state[settings.dimension_vector_estado - 2] = self.trackpos_rw

            return state, exit_flag
