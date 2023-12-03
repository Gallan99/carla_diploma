import glob
import os
import sys
sys.path.append("/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg")
import carla
import random
import time
import pdb
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner 




def main():
    actor_list = []

    try:
        #connect client with the carla server
        client = carla.Client('localhost', 2000) 
        client.set_timeout(100.0)
        #the world object has access to all the elements of the simulation.Load the town2
        world = client.load_world('Town02') 
        amap = world.get_map()
        sampling_resolution = 2
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        #generate starting and ending point and draw them
        point_start = carla.Transform(carla.Location(x=150.240021, y=191.770035, z=0.500000), carla.Rotation(pitch=0.000000, yaw=-0.000183, roll=0.000000))
        point_start2 = carla.Transform(carla.Location(x=34.700912, y=187.568146, z=1.000000), carla.Rotation(pitch=360.000000, yaw=-179.973694, roll=0.000000))
        point_end = carla.Location(x=195.00, y=200.090454, z=0.000000)
        world.debug.draw_string(point_start.location, 'X', draw_shadow=False,color=carla.Color(r=255, g=0, b=0), life_time=1000000,persistent_lines=True)
        world.debug.draw_string(point_end, 'X', draw_shadow=False,color=carla.Color(r=255, g=0, b=0), life_time=1000000,persistent_lines=True)
        #via blueprint library we can add new actors
        blueprint_library = world.get_blueprint_library()
        #choose a random vehicle from the library
        bp = blueprint_library.find('vehicle.audi.tt')
        bp2 = blueprint_library.find('vehicle.dodge.charger_2020') 
        #spawn the vehicles at the starting point that we choose
        car = world.spawn_actor(bp, point_start)
        car2 = world.spawn_actor(bp2, point_start2)
        actor_list.append(car)
        actor_list.append(car2)
        #generate more cars
        wp = amap.generate_waypoints(10)
        for i in range(0,len(wp)):
            npc_bp = random.choice(blueprint_library.filter('vehicle'))
            npc_car = world.try_spawn_actor(npc_bp, wp[i].transform)
            if npc_car is not None:
                npc_car.set_autopilot(True)
                actor_list.append(npc_car)

        #make the route that we want to execute and draw it
        route = grp.trace_route(point_start.location, point_end)
        for waypoint in route:
            world.debug.draw_string(waypoint[0].transform.location, 'O', draw_shadow=False,color=carla.Color(r=0, g=255, b=0), life_time=1000000,persistent_lines=True)
        #making an instance of basic agent 
        b_a_1=BasicAgent(car)
        b_a_2=BehaviorAgent(car2, behavior='aggressive')
        #import the route on the fuction
        b_a_1.set_global_plan(route)
        b_a_2.set_destination(wp[3].transform.location,point_start2)
        while True:
            car.apply_control(b_a_1.run_step())
            car2.apply_control(b_a_2.run_step())






        
        

        time.sleep(200)

    finally:
            print("end")
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            
        

if __name__ == '__main__':

    main()

