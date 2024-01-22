from layers import Conv1d, Res1d, Linear, LinearRes, Null
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from math import gcd
from utils import gpu, to_long,  Optimizer, StepLR
from CarlaDataset import CarlaDataset, collate_fn
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.actor_net = ActorNet()
        self.map_net = MapNet()
        self.lane_net = LaneNet()
        self.m2m = M2M()
        self.a2m = A2M()
        self.m2a = M2A()
        self.a2a = A2A()
        self.pred_net = PredNet()

    def forward(self, data):
        # construct actor feature
        actors, actor_idcs = actor_gather(gpu(data["feat"]))
        actor_ctrs = gpu(data["ctrs"])
        actors = self.actor_net(actors)

        # construct map features 1
        lanes, lane_idcs = lane_gather(gpu(data['nbr_waypoints']))
        lane_ctrs = gpu(data['nbr_waypoints'])
        lanes = self.map_net(lanes)
        
        
        # construct map features 2
        lane_lists = gpu(data['lane_list'])
        padded_maps_tensor = lane_normalization(lane_lists, max_length=1200)
        lanes_topo_feature = self.lane_net(padded_maps_tensor)

        # actor-map fusion cycle 
        lanes = self.m2m(lanes, lane_idcs, lanes_topo_feature)
        lanes = self.a2m(lanes, lane_idcs, lane_ctrs, actors, actor_idcs, actor_ctrs)
        actors = self.m2a(actors, actor_idcs, actor_ctrs, lanes, lane_idcs, lane_ctrs)
        actors = self.a2a(actors, actor_idcs, actor_ctrs)
        
        # prediction
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        

        return out



class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self):
        super(ActorNet, self).__init__()
        norm = "GN"
        ng = 1
        n_in = 7
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = 128
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, actors):
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out


def actor_gather(actors):
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs


# class MapNet(nn.Module):
#     def __init__(self):
#         super(MapNet, self).__init__()
#         input_size = 2
#         output_size = 128
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, output_size)
        
#     def forward(self, x):
#         x = x.float()
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class MapNet(nn.Module):
    def __init__(self):
        super(MapNet, self).__init__()
        input_size = 2
        output_size = 128
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)  
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)  
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.bn1(x) 
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x) 
        x = torch.relu(x)
        x = self.fc3(x)
        return x




def lane_gather(lanes):
    batch_size = len(lanes)
    num_lanes = [len(x) for x in lanes]
    lanes = torch.cat(lanes, 0)
    lane_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_lanes[i]).to(lanes.device)
        lane_idcs.append(idcs)
        count += num_lanes[i]
    return lanes, lane_idcs

# class LaneNet(nn.Module):
#     def __init__(self):
#         super(LaneNet, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.linear = nn.Linear(128, 128)

#     def forward(self, x):
#         # 输入形状: [32, 1200, 2]
#         x = x.permute(0, 2, 1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = torch.mean(x, dim=2)
#         x = x.permute(0, 1).contiguous()
#         x = self.linear(x)
#         x = x.unsqueeze(1) 
#         return x
    
    
class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64) 
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128) 
        self.linear = nn.Linear(128, 128)

    def forward(self, x):
        # 输入形状: [32, 1200, 2]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) 
        x = torch.relu(x) 
        x = torch.mean(x, dim=2)
        x = x.permute(0, 1).contiguous()
        x = self.linear(x)
        x = x.unsqueeze(1)
        return x    
    
    
    
def lane_normalization(lane_lists, max_length=1200):
    # 填充和裁剪地图数据
    padded_maps = []
    for lane_list in lane_lists:
        length = min(len(lane_list), max_length)
        padded_lane_list = lane_list[:length]
        padding = max_length - length
        padding_tensor = torch.zeros(padding, 2).cuda()
        padded_lane_list = torch.cat((padded_lane_list, padding_tensor), dim=0)
        padded_maps.append(padded_lane_list)

    padded_maps_tensor = torch.stack(padded_maps)
    return padded_maps_tensor



class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """
    def __init__(self) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1
        n_agt = 128
        n_ctx = 128
        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agts, agt_idcs, agt_ctrs, ctx, ctx_idcs, ctx_ctrs):
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts) 
            agts = self.relu(agts)    
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= 100.0

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        
        # 将输入张量和权重转换为相同的数据类型
        dist = self.dist(dist.to(agts.dtype))
        query = self.query(agts[hi])
        
        # 将输入张量和权重转换为相同的数据类型
        ctx = ctx[wi].to(agts.dtype)
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts


class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """
    def __init__(self):
        super(A2A, self).__init__()
        norm = "GN"
        ng = 1

        n_actor = 128
        n_map = 128

        att = []
        for i in range(2):
            att.append(Att())
        self.att = nn.ModuleList(att)

    def forward(self, actors, actor_idcs, actor_ctrs):
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
            )
        return actors
        

class A2M(nn.Module):
    def __init__(self):
        super(A2M, self).__init__()
        n_map = 128
        norm = "GN"
        ng = 1

        att = []
        for i in range(2):
            att.append(Att())
        self.att = nn.ModuleList(att)

    def forward(self, lanes, lane_idcs, lane_ctrs, actors, actor_idcs, actor_ctrs):
        for i in range(len(self.att)):
            feat = self.att[i](
                lanes,
                lane_idcs,
                lane_ctrs,
                actors,
                actor_idcs,
                actor_ctrs
            )

        return feat

# class M2M(nn.Module):
#     def __init__(self):
#         super(M2M, self).__init__()
#         self.linear1 = nn.Linear(256, 512)
#         self.linear2 = nn.Linear(512, 256)
#         self.linear3 = nn.Linear(256, 128)

#         # 初始化模型参数
#         for layer in [self.linear1, self.linear2, self.linear3]:
#             nn.init.xavier_uniform_(layer.weight)
#             nn.init.zeros_(layer.bias)

#     def forward(self, lanes, lane_idcs, lanes_topo_feature):
#         batch_size = len(lane_idcs)

#         lane_features = []
#         for i in range(batch_size):
#             len_maps = len(lane_idcs[i])
#             lane_feature = lanes_topo_feature[i].repeat(len_maps, 1)
#             lane_features.append(lane_feature)

#         lane_features = torch.cat(lane_features, dim=0)
#         lanes_concat = torch.cat((lanes, lane_features), dim=1)
        
#         x = F.relu(self.linear1(lanes_concat))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear3(x))

#         return x
        

class M2M(nn.Module):
    def __init__(self):
        super(M2M, self).__init__()
        self.linear1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512)  
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)  
        self.linear3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128) 

        # 初始化模型参数
        for layer in [self.linear1, self.linear2, self.linear3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, lanes, lane_idcs, lanes_topo_feature):
        batch_size = len(lane_idcs)

        lane_features = []
        for i in range(batch_size):
            len_maps = len(lane_idcs[i])
            lane_feature = lanes_topo_feature[i].repeat(len_maps, 1)
            lane_features.append(lane_feature)

        lane_features = torch.cat(lane_features, dim=0)
        lanes_concat = torch.cat((lanes, lane_features), dim=1)
        
        x = F.relu(self.bn1(self.linear1(lanes_concat)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))

        return x




class M2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """
    def __init__(self):
        super(M2A, self).__init__()
        norm = "GN"
        ng = 1

        n_actor = 128
        n_map = 128

        att = []
        for i in range(2):
            att.append(Att())
        self.att = nn.ModuleList(att)

    def forward(self, actors, actor_idcs, actor_ctrs, lanes, lane_idcs, lane_ctrs):
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                lanes,
                lane_idcs,
                lane_ctrs
            )
        return actors


class AttDest(nn.Module):
    def __init__(self):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1
        n_agt = 128
        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts, agt_ctrs, dest_ctrs):
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts


class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """
    def __init__(self):
        super(PredNet, self).__init__()
        norm = "GN"
        ng = 1
        n_actor = 128
        pred = []
        for i in range(6):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * 30),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest()
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )

    def forward(self, actors, actor_idcs, actor_ctrs):
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, 6)

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out


class PredLoss(nn.Module):
    def __init__(self):
        super(PredLoss, self).__init__()
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, gt_preds, has_preds):
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x[0].unsqueeze(0) for x in cls], 0)
        reg = torch.cat([x[0].unsqueeze(0) for x in reg], 0)
        gt_preds = torch.cat([x.unsqueeze(0) for x in gt_preds], 0).detach()
        has_preds = torch.cat([x.unsqueeze(0) for x in has_preds], 0).detach()
        has_preds = (has_preds == 1).detach()

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = 6, 30

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < 2.0).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > 0.2
        mgn = mgn[mask0 * mask1]
        mask = mgn < 0.2
        coef = 1.0
        loss_out["cls_loss"] += coef * (
            0.2 * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = 1.0
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out


    
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.pred_loss = PredLoss()

    def forward(self, out, data):
        data['gt_preds'] = [label[:, :2] for label in data['label']]
        data['has_preds'] = [label[:, 2] for label in data['label']]
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out    
    
    
    

class PostProcess(nn.Module):
    def __init__(self):
        super(PostProcess, self).__init__()

    def forward(self, out, data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        data["gt_preds"] = [label[:, :2] for label in data["label"]]
        data["has_preds"] = [label[:, 2] for label in data["label"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics, loss_out, post_out):
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss, cls, reg, ade1, fde1, ade, fde)
        )
        print()



def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, (1, 2))) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs



def get_model():
    net = Net().cuda()
    loss = Loss()
    post_process = PostProcess().cuda()
    params = net.parameters()
    opt = Optimizer(params)
    # opt = optim.Adam(net.parameters(), lr=5e-4)    
    return CarlaDataset, collate_fn, net, loss, post_process, opt