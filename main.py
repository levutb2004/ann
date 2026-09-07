import time
from models import *
from osgeo import gdal
import numpy as np
from sklearn import metrics
import os
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tools import *
import numpy_indexed as npi
import pandas as pd
import gc
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Program description')

# Add arguments
parser.add_argument('--ckp', type=str, default='None', help='Path to the model checkpoint file')
parser.add_argument('--ep', type=int, default=50000, help='Number of training epochs')
parser.add_argument('--mode', type=str,
                    choices=['train', 'train_tpu', 'map', 'eval'], default='train',
                    help='Program mode')
parser.add_argument('--type', type=str, default='both', help='Loss function type')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--dis', '-d', type=str, default='', help='Additional description')
parser.add_argument('--year', '-y', type=str, default='2019', help='Dataset year')
parser.add_argument('--savepath', '-s', type=str, default='None')

# Parse arguments
args = parser.parse_args()

args.map = "data_lib/" + \
    {'2021': "HongKong2021_FeatureCube.tif",
     '2019': 'VietNam2019_FeatureCube.tif'}[args.year]

if (args.mode == 'map') & (args.savepath == 'None'):
    raise ValueError('Please set the save path.')

# Use arguments
print("Checkpoint file path:", args.ckp, end='\n')
print("Training epochs:", args.ep, end='\n')
print("Program mode:", args.mode, end='\n')
print("Data type:", args.type, end='\n')
print("Learning rate:", args.lr, end='\n')
print("Additional description:", args.dis, end='\n')
print("Dataset year:", args.year, end='\n')


# load and split data HK
Dataset = gdal.Open(
    args.map).ReadAsArray()

# factors = tuple(factors)
DataFactors = Dataset[:-6]
DataFactors[DataFactors > 1e30] = 0
DataFactors[DataFactors <= -1e38] = 0
PopDensity = Dataset[(-4, -1), :, :]
Area = Dataset[(-5, -2), :, :]
RegionMask = Dataset[(-6, -3), :, :]
RegionMask = Dataset[(-6, -3), :, :]
del Dataset
gc.collect()
c, m, n = DataFactors.shape

'''For data in 2016, please add the following 3 patches'''
'''For data in 2016, please add the following 3 patches'''
'''For data in 2016, please add the following 3 patches'''
# RegionMask[1][RegionMask[0] == 800] = 214
# PopDensity[1][RegionMask[0] == 800] = 25.806874
# Area[1][Area[0] == 800] = 45.879248

mask = RegionMask[0] > 0

for i in range(c):
    band = DataFactors[i]

    # ---- min-max normalize ----
    valid_vals = band[mask]
    band_min = valid_vals.min()
    band_max = valid_vals.max()
    band_range = band_max - band_min
    if band_range == 0:
        band_range = 1e-8  # tránh chia 0 cho band hằng số

    band -= band_min
    band /= band_range

    # ---- z-score standardize ----
    valid_vals = band[mask]  # đã normalize, tính lại mean/std
    band_mean = valid_vals.mean()
    band_std = valid_vals.std()
    if band_std == 0:
        band_std = 1

    band -= band_mean
    band /= band_std

    del valid_vals
    gc.collect()

Norm_DataFactors = DataFactors  

Group_TPU = {}
Group_TPU['factor'] = group_aggregation(
    Norm_DataFactors, RegionMask[1], False)[1][1:]
Group_TPU['pop'] = group_aggregation_(
    PopDensity[1], RegionMask[1], False)[1][1:]


in_data_grid = torch.from_numpy(
    Norm_DataFactors.reshape((c, m*n)).T).float()

in_data_aggr = torch.from_numpy(Group_TPU['factor']).cuda().float()

Group_TPU['pop'] = np.log10(Group_TPU['pop']+1)

re_data_aggr_by_division = torch.from_numpy(
    Group_TPU['pop']).cuda().float().squeeze()

EPOCH = args.ep
LR = args.lr
CKP = args.ckp
l2_loss = nn.MSELoss(reduction='none')
model = Model_SI(Norm_DataFactors.shape[0]).cuda()
del DataFactors, Norm_DataFactors, mask
gc.collect()
if args.mode in ['train', 'train_tpu']:
    optimizer = optim.Adam(model.parameters())
if CKP != 'None':
    checkpoint = torch.load("checkpoints/"+CKP)
    model.load_state_dict(checkpoint['model'])
params_str = '_'.join([
    args.ckp,
    str(args.ep),
    args.mode,
    args.type,
    '{:.2e}'.format(args.lr),
    args.dis
])

def train_per_tpu():
    region_flat = RegionMask[1].ravel()

    unique_labels, inverse = np.unique(region_flat, return_inverse=True)
    del region_flat
    gc.collect()

    n_groups = len(unique_labels)

    order = np.argsort(inverse, kind='stable')
    sorted_inverse = inverse[order]
    del inverse
    gc.collect()

    group_boundaries = np.searchsorted(sorted_inverse, np.arange(n_groups + 1))
    del sorted_inverse
    gc.collect()

    group_pixel_idx = []
    for i in range(n_groups):
        if unique_labels[i] == 0:
            continue
        idx = order[group_boundaries[i]:group_boundaries[i+1]].copy()
        group_pixel_idx.append(torch.from_numpy(idx).long())

    del order, group_boundaries
    gc.collect()

    n_groups = len(group_pixel_idx)  # cập nhật lại sau khi loại nodata
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    model_flag = timestamp + '_' + params_str
    log_root = f"log/{model_flag}"
    print(log_root)
    summaryWriter = SummaryWriter(log_root)

    loss_bar = tqdm(range(1, EPOCH + 1))
    min_rmse = 1e7

    for epoch in loss_bar:
        perm = torch.randperm(n_groups)
        epoch_loss = 0.0
        for g in perm.tolist():
            pix_idx = group_pixel_idx[g]
            batch_x = in_data_grid[pix_idx].cuda()

            pred = model(batch_x).squeeze(-1)
            del batch_x
            gc.collect()
            delog_pred = torch.pow(10, pred)
            pred_sum = delog_pred.sum()
            pred_log_aggr = torch.log10(pred_sum)

            target = re_data_aggr_by_division[g]
            loss = l2_loss(pred_log_aggr, target).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_groups

        model.eval()
        rmse = validation(model)
        model.train()

        if min_rmse >= rmse.item():
            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, f'checkpoints/{model_flag}.pth')
            min_rmse = rmse.item()

        loss_bar.set_postfix({'loss': '%.5f' % avg_loss,
                               'RMSE': '%.3f' % rmse,
                               'min_rmse': '%.5f' % min_rmse})
        summaryWriter.add_scalar("loss", avg_loss, epoch)
        summaryWriter.add_scalar("rmse", rmse, epoch)
def train():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    model_flag = timestamp + '_' + params_str
    log_root = f"log/{model_flag}"
    print(log_root)
    summaryWriter = SummaryWriter(log_root)
    loss_bar = tqdm(range(1, EPOCH+1))
    min_rmse = 1e7
    for epoch in loss_bar:
        pop_p_grid = predict_in_batches(model,in_data_grid).reshape(m, n)
        delog_pop_p_grid = torch.pow(10, pop_p_grid)
        pop_p_grid_aggr_TPU = torch.log10(aggragate_torch(
            delog_pop_p_grid, torch.from_numpy(RegionMask[1]))[1:]).squeeze(-1)
        l_grid_tpu = l2_loss(pop_p_grid_aggr_TPU, re_data_aggr_by_division)
        loss_tpu = l_grid_tpu.mean()

        loss = loss_tpu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        rmse = validation(model)
        if min_rmse >= rmse.item():
            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, f'checkpoints/{model_flag}.pth')
            min_rmse = rmse
        model.train()
        loss_bar.set_postfix({'loss:': '%.5f' % loss.item(
        ), 'RMSE:': '%.3f' % rmse, 'min_rmse:': '%.5f' % min_rmse})
        summaryWriter.add_scalar("loss", loss.item(), epoch)
        summaryWriter.add_scalar("rmse", rmse, epoch)
        

def validation(model):
    with torch.no_grad():
        pop_p = predict_in_batches(model,in_data_grid).reshape((m, n))
        unit = 1
        population_potential_delog = np.power(10, pop_p)
        sum_potential_delog_by_org_untit = group_aggregation_(
            population_potential_delog, RegionMask[unit], keep_shape=True, method='sum')
        sum_population_by_org_untit = PopDensity[unit]*Area[unit]
        population_redistributed = population_potential_delog / \
            sum_potential_delog_by_org_untit * sum_population_by_org_untit
        # evaluation
        unit = 0
        population_redistributed_sum_by_su = group_aggregation_(
            population_redistributed, RegionMask[unit], False, method='sum')[1][1:]
        population_by_su_census_data = group_aggregation_(
            PopDensity[unit]*Area[unit], RegionMask[unit], False, method='mean')[1][1:]
        rmse = np.sqrt(metrics.mean_squared_error(
            population_by_su_census_data, population_redistributed_sum_by_su))
    return rmse

def copy_geoinfo_and_save_image(ref_img, array, save_path):

    ds = gdal.Open(ref_img)
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None
    metadata = {
        'count': 1,
        'height': array.shape[0],
        'width': array.shape[1],
        'dtype': str(array.dtype),
        'transform': geotransform,
        'crs': projection
    }
    array = np.array(array, dtype=metadata['dtype'])
    array = np.expand_dims(array, axis=0)

    if os.path.exists(save_path):
        raise ValueError('The save path already exists.')

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(
        save_path, metadata['width'], metadata['height'], metadata['count'], gdal.GDT_Float32)
    ds.SetGeoTransform(metadata['transform'])
    ds.SetProjection(metadata['crs'])
    ds.GetRasterBand(1).WriteArray(array[0])
    ds.FlushCache()
    ds = None
def predict_in_batches(model, data, batch_size=65536):
    preds = []
    n_samples = data.shape[0]

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = data[start:end].cuda()
            out = model(batch).squeeze(-1)
            preds.append(out.cpu().numpy())

    return np.concatenate(preds, axis=0)

def mapping():
    with torch.no_grad():
        pop_p = predict_in_batches(model,in_data_grid).reshape((m, n))

        unit = 0
        population_potential_delog = np.power(10, pop_p)

        sum_potential_delog_by_org_untit = group_aggregation_(
            population_potential_delog, RegionMask[unit], keep_shape=True, method='sum')
        sum_population_by_org_untit = PopDensity[unit]*Area[unit]

        population_redistributed = population_potential_delog / sum_potential_delog_by_org_untit * sum_population_by_org_untit
        out_map = population_redistributed
        out_map[out_map == -np.inf] = 0
        population_redistributed_sum_by_su = group_aggregation_(
            population_redistributed, RegionMask[0], False, method='sum')[1][1:]
        np.savetxt(f'map/{args.ckp}.csv',
                   population_redistributed_sum_by_su, delimiter=',')
        copy_geoinfo_and_save_image(args.map, out_map, args.savepath)



def evaluation(model):
    with torch.no_grad():
        model.eval()
        data = in_data_grid
        pop_p = model(
            data, train=True).squeeze(-1).cpu().numpy().reshape((m, n))
        unit = 1
        population_potential_delog = np.power(10, pop_p)
        sum_potential_delog_by_org_untit = group_aggregation_(
            population_potential_delog, RegionMask[unit], keep_shape=True, method='sum')
        sum_population_by_org_untit = PopDensity[unit]*Area[unit]
        population_redistributed = population_potential_delog / \
            sum_potential_delog_by_org_untit * sum_population_by_org_untit
        # evaluation
        unit = 0
        population_redistributed_sum_by_su = group_aggregation_(
            population_redistributed, RegionMask[unit], False, method='sum')[1][1:]
        population_by_su_census_data = group_aggregation_(
            PopDensity[unit]*Area[unit], RegionMask[unit], False, method='mean')[1][1:]
        rmse = np.sqrt(metrics.mean_squared_error(
            population_by_su_census_data, population_redistributed_sum_by_su))
        mae = metrics.mean_absolute_error(
            population_by_su_census_data, population_redistributed_sum_by_su)
        rmse_percent = rmse/population_by_su_census_data.mean()
        r2 = metrics.r2_score(population_by_su_census_data,
                              population_redistributed_sum_by_su)
        ckp_dis = ','.join(args.ckp.split('_'))
        print(f'{ckp_dis},{rmse},{mae},{rmse_percent},{r2}')


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'train_tpu':
        train_per_tpu()
    elif args.mode == 'map':
        mapping()
    else:
        evaluation(model)
