"""models.py contains the neural symbolic models"""
#  -*- coding: utf-8 -*-

import io
from pathlib import Path
from typing import Dict, List, Tuple

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz import scenario_visualization
from av2.map.map_api import ArgoverseStaticMap
import ltn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as img
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ltnModules import BasicModule, yield_human
from utils.baselineConfig import RAW_DATA_FORMAT
from utils.socialFeaturesUtils import SocialFeaturesUtils

OBS_LEN: int = 50
PRED_LEN: int = 60


class ScenarioDataset(Dataset):
    def __init__(self, root_dir: Path):
        """
        Args:
            root_dir: Path to local directory where Argoverse scenarios are stored.
        """
        self.all_scenario_files = sorted(root_dir.iterdir())
        self.preprocessor = SocialFeaturesUtils()

    def __len__(self):
        return len(self.all_scenario_files)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                             torch.FloatTensor, torch.FloatTensor, np.ndarray, torch.Tensor]:
        # The scenario path corresponding to idx.
        scenario_path = self.all_scenario_files[idx]
        # Obtain the file name without suffix.
        scenario_id = scenario_path.stem
        # The corresponding json file of HD map.
        static_map_path = scenario_path / f"log_map_archive_{scenario_id}.json"
        # The corresponding parquet file for all tracks which are associated with the scenario.
        tracks_path = scenario_path / f"scenario_{scenario_id}.parquet"
        # Argoversescenario instance.
        scenario = scenario_serialization.load_argoverse_scenario_parquet(tracks_path)
        # ALL tracks associated with scenario in dataframe form.
        df_tracks = scenario_serialization._convert_tracks_to_tabular_format(scenario.tracks)
        # Obtain the focal track.
        focal_id = scenario.focal_track_id
        focal_track = df_tracks[df_tracks["track_id"] == focal_id]
        # Obtain other tracks except the focal track.
        other_tracks = df_tracks[df_tracks["track_id"] != focal_id]
        focal_track_obs, social_tracks_obs, focal_track_gt = \
            self.preprocessor.compute_social_features(other_tracks,
                                                      focal_track.values,
                                                      OBS_LEN,
                                                      OBS_LEN + PRED_LEN,
                                                      RAW_DATA_FORMAT)
        # Loading argoversestaticmap.
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        _, ax = plt.subplots()
        scenario_visualization._plot_static_map_elements(static_map)
        plt.gca().set_aspect("equal", adjustable="box")
        # Minimize plot margins and make axes invisible.
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.show()
        # Save plotted frame to in-memory buffer.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        frame = img.open(buf)
        # Convert to numpy array.
        static_map_array = np.asarray(frame)/255

        desired_attr = [
            RAW_DATA_FORMAT["position_x"], RAW_DATA_FORMAT["position_y"],
            RAW_DATA_FORMAT["heading"], RAW_DATA_FORMAT["velocity_x"], RAW_DATA_FORMAT["velocity_y"]
        ]
        focal_object_type = focal_track_obs[:, :, RAW_DATA_FORMAT["object_type"]][:, 0]
        social_object_type = social_tracks_obs[:, :, RAW_DATA_FORMAT["object_type"]][:, 0]
        object_type = np.concatenate((focal_object_type, social_object_type))

        focal_obs = focal_track_obs[:, :, desired_attr]
        social_obs = social_tracks_obs[:, :, desired_attr]
        focal_gt = focal_track_gt[:, :, desired_attr]
        obs = torch.cat([torch.tensor(focal_obs), torch.tensor(social_obs)], dim=0) # 原始数据raw data的合并

        return (
            torch.FloatTensor(focal_obs.astype(float)),  # 1*50*5
            torch.FloatTensor(social_obs.astype(float)),  # n-1*50*5
            torch.FloatTensor(static_map_array).permute(2, 0, 1).contiguous(),  # 480*640*4
            torch.FloatTensor(focal_gt.astype(float)),  # 1*60*5
            object_type,  # n
            obs
        )


class LogicNet(nn.Module):
    def __init__(self):
        super(LogicNet, self).__init__()
        # Map size B*4*480*640
        self.map_encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=11, stride=4),  # B*8*118*158
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # B*8*58*78
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1),  # B*16*54*74
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # B*16*26*36
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),  # B*32*24*34
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # B*32*11*16
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),  # B*64*9*14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # B*64*4*6
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),  # B*128*2*4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # B*128*1*2
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        # Input coordinates, velocity, headings.
        self.track_encoder = nn.ModuleList([
            nn.Linear(5, 128), # 用linear扩展size到128
            nn.GRU(input_size=128, hidden_size=128, batch_first=True),
        ])
        # Gaussian distribution for Latent variable.
        self.gpl = nn.Sequential(
            nn.Linear(384, 256),
            nn.Tanh(),
        )
        # Linear layer for deterministic inference. cancat vectors
        self.dmi = nn.Sequential(
            nn.Linear(392, 128),
            nn.Tanh(),
        )

        # GRU decoder = linear layer + GRUcell(一个时间点一个时间点的，因为预测的轨迹是按时间点预测的) 152-162 prediction
        # Linear layer for decoder output
        self.do = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        # Decode future tracks.
        self.decoder = nn.GRUCell(input_size=128, hidden_size=128)

        # Output the Gaussian process for coordinates(x, y). 用高斯分布sample出预测的坐标
        self.gp = nn.Linear(128, 4)

        # ltn predicates.
        self.ltn_pass = ltn.Predicate(model=BasicModule(256, 128, 1))
        self.ltn_follow = ltn.Predicate(model=BasicModule(256, 128, 1))
        self.ltn_yield = ltn.Predicate(model=BasicModule(256, 128, 1)) # 188line 2个输入量，focal+other=256
        self.ltn_influence = ltn.Predicate(model=BasicModule(256, 128, 1))
        # by default, SatAgg uses the pMeanError.
        self.sat_agg = ltn.fuzzy_ops.SatAgg()

    def forward(self, x, y):
        tracks, av2_map = x, y # 传入 tracks and map
        _shape = tracks.shape  # B*n*50*5 tracks原始大小
        map_f = self.map_encoder(av2_map).view(_shape[0], -1)  # B*128 map features
        # track encoder, extract tracks features
        tracks_emb = self.track_encoder[0](tracks)  # B*n*50*128 先经过线性层，变成想要的size
        tracks_h = [] # h = hidden state 包含了所有timesteps的特征信息（只有RNN里的hidden state有自己的含义，其他都是连阶层的传递量）。只要h，不要output
        # 提取所有agent的tracks features
        for i in range(_shape[1]):
            _, track_h = self.track_encoder[1](tracks_emb[:, i, :, :])  # GRU,没有了时间维度，变成三维，1*B*128 fix size of hidden state(128自己设定的)
            tracks_h.append(track_h)
        tracks_h = torch.cat(tracks_h, dim=0) # cancate all agents' features n*B*128
        tracks_h = tracks_h.permute(1, 0, 2).contiguous() # use permute to exchange dimension to B*n*128
        focal_f = tracks_h[:, 0, :].unsqueeze(1) # B*n*128。n= 0，97line定义，第一个一定是focal
        other_f = tracks_h[:, 1:, :]
        # input features to ltn
        ltn_yield = self.ltn_yield.model(focal_f, other_f).squeeze()  # B*n-1 line167,维度是1就去掉，为了与后续的其他操作维度一致，所以size-1。将focal和other作为输入量输入ltn
        ltn_influence = self.ltn_influence.model(focal_f, other_f)  # B*n-1*1
        fused_others = torch.sum(other_f*F.softmax(ltn_influence, dim=1), dim=1)  # B*128 将interact features加权几何
        gaussian = self.gpl(torch.cat([focal_f.squeeze(), map_f, fused_others], dim=-1)) # concate features,
        mu, logvar = gaussian[:, :128], gaussian[:, 128:] # sampled variable
        z = sample_z(mu, logvar)
        code = self.dmi(torch.cat([focal_f.squeeze(), map_f, z, ltn_yield, ltn_influence.squeeze()], dim=-1))  # B*128 # 把所有的vectors concate，缩放到128size

        # prediction，将数据输入decoder中，decoder中RNN，每个时间点都进行预测，最终得到路径,line 152-162
        h = torch.zeros(code.shape) # 因为decoder有两个输入（x = code，hidden state = h = 0),line 200
        outputs = []
        # prediction length = 60, 预测60个点
        for _ in range(PRED_LEN):
            h = self.decoder(code, h)
            code = self.do(h)
            x = self.gp(h)
            xz = sample_z(x[:, :2], x[:, 2:])
            outputs.append(xz.unsqueeze(1))

        pred_traj = torch.cat(outputs, dim=1)
        return pred_traj

    def ltn_loss(self, other: torch.Tensor, focal: torch.Tensor) -> torch.FloatTensor:
        # compute ltn loss. tutorial
        loss = 1. - self.sat_agg(
            yield_human(other, focal, self.ltn_influence, self.ltn_yield)
        )
        return loss


def sample_z(mu, logvar, train=True):
    # if training models then sample from Gaussian, otherwise output mu. 采样方式
    if train:
        eps = torch.randn(mu.shape)
    else:
        eps = 0

    return mu + torch.exp(logvar/2)*eps


if __name__ == "__main__":
    from torchsummary import summary
    #dataset = ScenarioDataset(Path("/Volumes/TOSHIBA EXT/Argoverse2/train"))
    #a, b, c, d, e = dataset[0]

    model = LogicNet()
    summary(model, [torch.ones((2, 5, 50, 5)), torch.ones((2, 4, 480, 640))])
    # Batchsize = 2, nums of agents = 5. timesteps = 50, desire attr = 5 (86 line)

