import os
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm




def read_and_concatenate_files(folder_path):
    trajectory_data = pd.DataFrame()

    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        if os.path.isfile(filepath):  
            df = pd.read_csv(filepath, names=['vehicle_id', 'timestamp', 'x', 'y', 'traffic_flag', 'traffic_state'])
            trajectory_data = pd.concat([trajectory_data, df], ignore_index=True)

    return trajectory_data



def filter_nearby_vehicles(target_id, target_vehicle, grouped):
    target_area_vehicles = pd.DataFrame()

    for timestamp, group in grouped:
        # 将目标车辆的信息加入 target_area_vehicles
        target_info = target_vehicle[target_vehicle['timestamp'] == timestamp]
        target_area_vehicles = target_area_vehicles.append(target_info, ignore_index=True)

        # 确保目标信息非空
        if not target_info.empty:
            x_target, y_target = target_info['x'].values[0], target_info['y'].values[0]

            # 计算每行的距离
            group['distance_to_target'] = np.sqrt((group['x'] - x_target) ** 2 + (group['y'] - y_target) ** 2)

            # 筛选条件：vehicle_id不等于target_id且距离小于30
            filtered_group = group[(group['vehicle_id'] != target_id) & (group['distance_to_target'] < 30)]
            target_area_vehicles = target_area_vehicles.append(filtered_group[['vehicle_id', 'timestamp', 'x', 'y', 'traffic_flag', 'traffic_state']], ignore_index=True)
    return target_area_vehicles



def process_vehicle_data(target_id, target_group, filtered_df, all_timestamps):
    filtered_df['frame_exists'] = 1
    grouped_by_vehicle = filtered_df.groupby('vehicle_id')

    processed_data = []
    all_data_template = pd.DataFrame({'timestamp': all_timestamps})

    for vehicle_id, group in grouped_by_vehicle:
        if vehicle_id == target_id:
            target_group = group
            continue

        all_data = all_data_template.copy()
        all_data['vehicle_id'] = vehicle_id
        merged_data = pd.merge(all_data, group, on=['timestamp', 'vehicle_id'], how='left').fillna(0)
        processed_data.append(merged_data)

    all_data = all_data_template.copy()
    all_data['vehicle_id'] = target_id
    merged_target_data = pd.merge(all_data, target_group, on=['timestamp', 'vehicle_id'], how='left').fillna(0)
    processed_data.insert(0, merged_target_data)

    return processed_data



def slice_data_frames(processed_data, slice_size=50, duration=600):
    sliced_data = []
    for t in range(duration // slice_size):
        start = t * slice_size
        end = (t + 1) * slice_size
        sliced_df_list = [vehicle_df.iloc[start:end].copy() for vehicle_df in processed_data]
        sliced_data.append(sliced_df_list)

    return sliced_data



def construct_features_and_labels(sliced_data, traj_data):
    # 对每个样本进行处理
    for sliced_df_list in sliced_data:
        # 提取标签（后30帧）
        label = sliced_df_list[0][['x', 'y', 'frame_exists']][20:].values.tolist()

        # 构建特征（前20帧）
        feature_list = []
        for vehicle_df in sliced_df_list:
            row_19 = vehicle_df.iloc[19]
            if row_19['frame_exists'] == 1:
                features = vehicle_df[['x', 'y', 'frame_exists', 'traffic_flag', 'traffic_state']][:20].values.tolist()
                feature_list.append(features)
        traj_data.append([feature_list, label])
    
    return traj_data


if __name__ == '__main__':
    folder_path = 'trajectories_trafficLight'
    traj_data = []
    all_timestamps = range(600)
    for root, dirs, files in os.walk(folder_path):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            print(f"当前文件夹: {folder_path}")

            trajectory_data = read_and_concatenate_files(folder_path)

            # 按时间戳和车辆ID排序， trajectory_data是600帧的全部轨迹
            trajectory_data = trajectory_data.sort_values(by=['timestamp', 'vehicle_id'])
            grouped_vehicle = trajectory_data.groupby('vehicle_id')
            grouped_timestamp = trajectory_data.groupby('timestamp')

            # 顺序选择目标车辆，并按照时间戳分组，每个时间戳下的others车辆按距离筛选
            for vehicle_id, group in tqdm(grouped_vehicle):
                target_group = group
                target_others_info_df = filter_nearby_vehicles(vehicle_id, target_group, grouped_timestamp)
                processed_data = process_vehicle_data(vehicle_id, target_group, target_others_info_df, all_timestamps)
                sliced_data = slice_data_frames(processed_data)
                traj_data = construct_features_and_labels(sliced_data, traj_data)

    # traj_data存储的是所有样本，每个样本traj_data[i] = [feature, label]
    traj_data_save_path = 'trajectories_trafficLight.pkl'
    with open(traj_data_save_path, 'wb') as f:
        pickle.dump(traj_data, f)


