import carla
import random
import csv
import time
import os
from datetime import datetime

def record_vehicle_trajectories(world, vehicle_blueprints, output_dir, duration, map_name, vehicles):
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(output_dir, f'{map_name}_duration_{duration}_{current_time}')
    os.makedirs(output_dir, exist_ok=True)

    spawn_points = world.get_map().get_spawn_points()

    for _ in range(100):
        blueprint = random.choice(vehicle_blueprints)
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(blueprint, spawn_point)

        if vehicle is not None:
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)

    trajectories = {vehicle.id: [] for vehicle in vehicles}

    print(f"Recording {duration * 0.1} seconds of vehicle trajectories...")

    start_time = time.time()
    frame = 0  # 添加时间戳
    interval = 0.1

    while frame < duration:
        loop_start_time = time.time()

        for vehicle in vehicles:
            location = vehicle.get_location()
            vehicle_id = vehicle.id
            x, y = location.x, location.y
            trajectories[vehicle_id].append((vehicle_id, frame, x, y))

        world.tick()

        loop_end_time = time.time()
        loop_elapsed_time = loop_end_time - loop_start_time
        sleep_time = max(0, interval - loop_elapsed_time)

        time.sleep(sleep_time)
        frame += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time elapsed: {elapsed_time} seconds")



    # 存储轨迹数据
    for vehicle_id, trajectory in trajectories.items():
        filename = os.path.join(output_dir, f'vehicle_{vehicle_id}_trajectory.csv')
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(trajectory)


if __name__ == '__main__':
    vehicles = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)

    record_duration = 600
    map_name = 'Town05'

    world = client.load_world(map_name)
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
    output_dir = 'trajectories'
    record_vehicle_trajectories(world, vehicle_blueprints, output_dir, record_duration, map_name, vehicles)
