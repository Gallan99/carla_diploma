
import glob
import os
import sys
sys.path.append("/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg")
import carla
import random
import time


from agents.navigation.global_route_planner import GlobalRoutePlanner





def main():
    actor_list = []
    try:
        route_list=[]
        #connect client with the carla server
        client = carla.Client('localhost', 2000) 
        client.set_timeout(100.0)
        #the world object has access to all the elements of the simulation.Load the town2
        world = client.load_world('Town02') 
        amap = world.get_map()
        sampling_resolution = 2
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        point_start = carla.Location(x=45.850761, y=198.209839, z=0.000000)
        point_end = carla.Location(x=189.688553, y=200.090454, z=0.000000)
        world.debug.draw_string(point_start, 'X', draw_shadow=False,color=carla.Color(r=255, g=0, b=0), life_time=1000000,persistent_lines=True)
        world.debug.draw_string(point_end, 'X', draw_shadow=False,color=carla.Color(r=255, g=0, b=0), life_time=1000000,persistent_lines=True)
        route = grp.trace_route(point_start, point_end)
        print(len(route))
        for waypoint in route:
            print(waypoint[1])
            route_list.append(waypoint[0])
            world.debug.draw_string(waypoint[0].transform.location, 'O', draw_shadow=False,color=carla.Color(r=0, g=255, b=0), life_time=1000000,persistent_lines=True)
        for i in route_list:
            print(i.transform.location)
        

        time.sleep(20)

    finally:
            print("end")
            
        

if __name__ == '__main__':

    main()
