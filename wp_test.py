import glob
import os
import sys

sys.path.append("/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg")
import carla
import random
import time
import pdb


def main():
    actor_list = []

    try:
        next_point = []
        # connect client with the carla server
        client = carla.Client('localhost', 2000)
        client.set_timeout(100.0)
        # the world object has access to all the elements of the simulation.Load the town2
        world = client.load_world('Town02')
        map = world.get_map()
        # Generate random waypoints with a 200cm distance between them
        wp = map.generate_waypoints(200)
        # print the coordinates x,y,z and draw them on the map with red colour.Using next method to take the next
        # available waypoint with a distance 1cm
        for i in range(0, len(wp)):
            print(str(i) + ")" + str(wp[i].transform.location))
            world.debug.draw_string(wp[i].transform.location, 'X', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=10000, persistent_lines=True)
            next_point.append(wp[i].next(1))
        # print the next avaible waypoint from the list next_point
        for i in next_point:
            for j in i:
                print(j.transform.location)
                world.debug.draw_string(j.transform.location, 'X', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=10000, persistent_lines=True)

        time.sleep(20)

    finally:
        print("end")


if __name__ == '__main__':
    main()
