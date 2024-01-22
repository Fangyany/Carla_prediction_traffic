import os

import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
from utils import Logger, load_pretrain, gpu

from CarlaDataset import from_numpy, worker_init_fn
from Net import get_model
from sklearn.model_selection import train_test_split


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


torch.cuda.set_device(3)

torch.autograd.set_detect_anomaly(True)



config = dict()
config["epoch"] = 0
config["batch_size"] = 32
config["save_dir"] = "./weight_and_log_trafficLight_mapdict_yaw"
config["save_freq"] = 1
config["num_epochs"] = 40
config["display_freq"] = 10
config["test_freq"] = 1


def main():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    Dataset, collate_fn, net, loss, post_process, opt = get_model()

    # 加载预训练权重
    # ckpt_path = ''
    # if not os.path.isabs(ckpt_path):
    #     ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    # ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    # load_pretrain(net, ckpt["state_dict"])

    # config["epoch"] = ckpt["epoch"]
    # opt.load_state_dict(ckpt["opt_state"])



    # Create log and copy all code
    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sys.stdout = Logger(log)

    src_dirs = [root_path]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
    # dataset = Dataset()
    # train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # 保存训练集和测试集
    # with open('train_data_yaw.pkl', 'wb') as train_file:
    #     pickle.dump(train_data, train_file)

    # with open('test_data_yaw.pkl', 'wb') as test_file:
    #     pickle.dump(test_data, test_file)
        
    # 加载训练集和测试集
    with open('train_data_yaw.pkl', 'rb') as train_file:
        train_data = pickle.load(train_file)

    with open('test_data_yaw.pkl', 'rb') as test_file:
        test_data = pickle.load(test_file)

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )



    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in tqdm(range(remaining_epochs)):
        train(epoch + i, config, train_loader, net, loss, post_process, opt, test_loader)




def train(epoch, config, train_loader, net, loss, post_process, opt, test_loader=None):
    net.train()
    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(np.ceil(config["display_freq"] * num_batches))
    test_iters = int(np.ceil(config["test_freq"] * num_batches))
    
    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(train_loader):
        epoch += epoch_per_batch
        data = dict(data)
        output = net(data)
        
        loss_out = loss(output, data)
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)
        opt.zero_grad()
        loss_out["loss"].backward(retain_graph=True)
        lr = opt.step(epoch)
        
        num_iters = int(np.round(epoch * num_batches))
        if num_iters % save_iters == 0 or epoch >= config["num_epochs"]:
            save_ckpt(net, opt, config["save_dir"], "net", epoch)

        # if num_iters % display_iters == 0:
        #     dt = time.time() - start_time
        #     # metrics = sync(metrics)
        #     post_process.display(metrics, dt, epoch, lr)
        #     start_time = time.time()
        #     metrics = dict()

        if num_iters % test_iters == 0:
            test(test_loader, net, loss, post_process, epoch)
            return




def test(data_loader, net, loss, post_process, epoch):
    net.eval()
    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    # metrics = sync(metrics)
    # if hvd.rank() == 0:
    post_process.display(metrics, dt, epoch)
    net.train()



def save_ckpt(net, opt, save_dir, model_name, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = model_name + "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.state_dict()},
        os.path.join(save_dir, save_name),
    )
    
if __name__ == "__main__":
    main()