import carla
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def get_neighboring_waypoints(waypoint):
    left_waypoint = waypoint.get_left_lane()
    right_waypoint = waypoint.get_right_lane()
    previous_waypoint = waypoint.previous(5) if waypoint.previous(1) else None
    next_waypoint = waypoint.next(5) if waypoint.next(1) else None
    return {'left': left_waypoint, 'right': right_waypoint, 'previous': previous_waypoint, 'next': next_waypoint}


def generate_map_data(world_map, granularity=1.0):
    data_list = []

    for waypoint in world_map.generate_waypoints(granularity):
        x = waypoint.transform.location.x
        y = waypoint.transform.location.y
        road_id = waypoint.road_id
        lane_id = waypoint.lane_id
        s = waypoint.s
        data_list.append([x, y, road_id, lane_id, s])

    map_df = pd.DataFrame(data_list, columns=['x', 'y', 'road_id', 'lane_id', 's'])
    return map_df


def extract_and_append_road_data(map_df, waypoint, road_data_list, label_prefix=''):
    if waypoint is not None:
        lane_id, road_id = waypoint.lane_id, waypoint.road_id
        road_data = map_df[(map_df['lane_id'] == lane_id) & (map_df['road_id'] == road_id)]
        # plt.plot(road_data['x'], road_data['y'], '-o', label=f"{label_prefix} Lane {lane_id}, Road {road_id}")
        road_data_list.append(road_data)


def calculate_midpoint(waypoint_tuple):
    waypoint_start, waypoint_end = waypoint_tuple
    x_start, y_start = waypoint_start.transform.location.x, waypoint_start.transform.location.y
    x_end, y_end = waypoint_end.transform.location.x, waypoint_end.transform.location.y
    mid = ((x_start + x_end) / 2, (y_start + y_end) / 2)
    return mid

def process_waypoint_tuple(map_df, waypoint_tuple, map_dict):
    waypoint_start, waypoint_end = waypoint_tuple

    road_data_list = []
    # Get neighboring waypoints
    neighboring_waypoints_start = get_neighboring_waypoints(waypoint_start)
    neighboring_waypoints_end = get_neighboring_waypoints(waypoint_end)

    # extract road data for start and end waypoints
    extract_and_append_road_data(map_df, waypoint_start, road_data_list, label_prefix='Start')
    extract_and_append_road_data(map_df, waypoint_end, road_data_list, label_prefix='End')

    # extract road data for neighboring waypoints
    for direction, waypoints in neighboring_waypoints_start.items():
        if direction == 'previous' and waypoints:
            for i, wp in enumerate(waypoints):
                extract_and_append_road_data(map_df, wp, road_data_list, label_prefix=f"{direction.capitalize()} {i+1} of Start")
        elif direction == 'next' and waypoints:
            for i, wp in enumerate(waypoints):
                extract_and_append_road_data(map_df, wp, road_data_list, label_prefix=f"{direction.capitalize()} {i+1} of Start")
        elif direction == 'left' and waypoints:
            extract_and_append_road_data(map_df, waypoints, road_data_list, label_prefix=f"{direction.capitalize()} {1} of Start")
        elif direction == 'right' and waypoints:
            extract_and_append_road_data(map_df, waypoints, road_data_list, label_prefix=f"{direction.capitalize()} {1} of Start")
    
    for direction, waypoints in neighboring_waypoints_end.items():
        if direction == 'previous' and waypoints:
            for i, wp in enumerate(waypoints):
                extract_and_append_road_data(map_df, wp, road_data_list, label_prefix=f"{direction.capitalize()} {i+1} of End")
        elif direction == 'next' and waypoints:
            for i, wp in enumerate(waypoints):
                extract_and_append_road_data(map_df, wp, road_data_list, label_prefix=f"{direction.capitalize()} {i+1} of End")
        elif direction == 'left' and waypoints:
            extract_and_append_road_data(map_df, waypoints, road_data_list, label_prefix=f"{direction.capitalize()} {1} of End")
        elif direction == 'right' and waypoints:
            extract_and_append_road_data(map_df, waypoints, road_data_list, label_prefix=f"{direction.capitalize()} {1} of End")
    
    mid = calculate_midpoint(waypoint_tuple)
    if mid not in map_dict:
        map_dict[mid] = road_data_list
    else:
        map_dict[mid] += road_data_list

    return map_dict


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    map_name = 'Town05'
    world = client.load_world(map_name)
    world_map = world.get_map()

    waypoint_tuple_list = world_map.get_topology()

    # map_dict的key是midpoint，value是road_data_list=[road_data_df1, road_data_df2, ...]
    map_dict = {}
    map_df = generate_map_data(world_map)
    for waypoint_tuple in waypoint_tuple_list:
        map_dict = process_waypoint_tuple(map_df, waypoint_tuple, map_dict)

    save_path = 'map_dict.pkl'
    with open(save_path, 'wb') as file:
        pickle.dump(map_dict, file)


        







