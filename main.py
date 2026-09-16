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
from scipy import ndimage
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Program description')

# Add arguments
parser.add_argument('--ckp', type=str, default='None', help='Path to the model checkpoint file')
parser.add_argument('--ep', type=int, default=50000, help='Number of training epochs')
parser.add_argument('--mode', type=str,
                    choices=['train', 'train_tpu', 'map', 'map_interval', 'eval'], default='train',
                    help='Program mode')
parser.add_argument('--nsamples', type=int, default=100, help='Number of posterior/MC Dropout forward passes for mapping_interval')
parser.add_argument('--modeltype', type=str, choices=['mc_dropout', 'bnn', 'si', 'ep', 'pdp', 'npdp'], default='mc_dropout',
                    help='Which model architecture to use')
parser.add_argument('--klweight', type=float, default=1e-4, help='Weight of the KL term in the ELBO loss (bnn only)')
parser.add_argument('--kmembers', type=int, default=5, help='Number of ensemble members (ep only)')
parser.add_argument('--wtaeps', type=float, default=0.1,
                    help='Relaxed-WTA weight for non-winner members (ep only); 0 = hard WTA')
parser.add_argument('--pdpdist', type=str, choices=['gaussian', 'student_t'], default='gaussian',
                    help='Distribution family predicted by PDP (pdp only)')
parser.add_argument('--quantiles', type=float, nargs='+',
                    default=[0.025, 0.25, 0.5, 0.75, 0.975],
                    help='Quantile levels predicted by NPDP (npdp only)')
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

os.makedirs('map', exist_ok=True)
os.makedirs('log', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

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
_model_ctor = {'si': Model_SI, 'mc_dropout': Model_MC_Dropout, 'bnn': Model_BNN}
if args.modeltype == 'ep':
    model = Model_EP(Norm_DataFactors.shape[0], k=args.kmembers).cuda()
elif args.modeltype == 'pdp':
    model = Model_PDP(Norm_DataFactors.shape[0], distribution=args.pdpdist).cuda()
elif args.modeltype == 'npdp':
    model = Model_NPDP(Norm_DataFactors.shape[0], quantile_levels=tuple(args.quantiles)).cuda()
else:
    model = _model_ctor[args.modeltype](Norm_DataFactors.shape[0]).cuda()
is_bnn = isinstance(model, Model_BNN)
is_ep = isinstance(model, Model_EP)
is_pdp = isinstance(model, Model_PDP)
is_npdp = isinstance(model, Model_NPDP)
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

def wta_loss(pred_log_aggr_k, target, eps):
    """Relaxed Winner-Takes-All loss cho Ensemble Prediction.
    pred_log_aggr_k: (k,) log-tổng dự đoán của từng thành viên cho 1 group.
    target: scalar log-tổng thật (census, đã log10).
    Chỉ thành viên gần target nhất ('winner') được full gradient; các
    thành viên còn lại chỉ nhận trọng số eps (0 = hard WTA, giữ eps>0 để
    tránh 'dead' members không bao giờ được cập nhật)."""
    per_member = l2_loss(pred_log_aggr_k, target.expand_as(pred_log_aggr_k))
    winner = torch.argmin(per_member.detach())
    weights = torch.full_like(per_member, eps)
    weights[winner] = 1.0
    return (per_member * weights).sum() / (1.0 + eps * (per_member.numel() - 1))


def predict_in_batches_ensemble(model, data, batch_size=65536):
    """Giống predict_in_batches nhưng giữ nguyên toàn bộ chiều output
    (k thành viên của Model_EP, [mu, sigma, ...] của Model_PDP, m quantile
    của Model_NPDP...), trả về mảng (N, C) thay vì (N,)."""
    preds = []
    n_samples = data.shape[0]
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = data[start:end].cuda()
            out = model(batch)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_point_in_batches(model, data, batch_size=65536):
    """Trả về một ước lượng điểm (point estimate) dạng (N,) cho MỌI loại
    model, dùng thống nhất cho validation()/mapping()/evaluation():
    - SI / MC Dropout / BNN: giá trị output trực tiếp (đã có 1 chiều)
    - EP: trung bình của k thành viên
    - PDP: mu (tham số vị trí của phân phối)
    - NPDP: quantile gần 0.5 nhất trong quantile_levels (xấp xỉ median)
    """
    preds = []
    n_samples = data.shape[0]
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = data[start:end].cuda()
            out = model(batch)
            if is_ep:
                point = out.mean(dim=-1)
            elif is_pdp:
                point = out[:, 0]
            elif is_npdp:
                mid = torch.argmin(torch.abs(model.quantile_levels - 0.5)).item()
                point = out[:, mid]
            else:
                point = out.squeeze(-1)
            preds.append(point.cpu().numpy())
    return np.concatenate(preds, axis=0)

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

            pred = model(batch_x)
            del batch_x
            gc.collect()

            target = re_data_aggr_by_division[g]

            if is_ep:
                delog_pred = torch.pow(10, pred)
                pred_sum = delog_pred.sum(dim=0)          # (k,)
                pred_log_aggr = torch.log10(pred_sum)      # (k,)
                loss = wta_loss(pred_log_aggr, target, args.wtaeps)
            elif is_pdp:
                mu, sigma = pred[:, :1], pred[:, 1:2]
                mu_sum = torch.pow(10, mu).sum(dim=0)              # (1,)
                pred_log_aggr = torch.log10(mu_sum).squeeze(-1)     # scalar
                # Xấp xỉ: tổng hợp sigma bằng RMS trên toàn bộ pixel của group
                # (không có ground-truth pixel-level nên không thể lan truyền
                # phương sai một cách chặt chẽ; đây là ước lượng đại diện).
                sigma_log_aggr = torch.sqrt((sigma ** 2).mean()) + 1e-6
                loss = (0.5 * ((pred_log_aggr - target) / sigma_log_aggr) ** 2
                        + torch.log(sigma_log_aggr))
            elif is_npdp:
                delog_q = torch.pow(10, pred)                # (n_pixels, m)
                q_sum = delog_q.sum(dim=0)                     # (m,)
                pred_log_aggr = torch.log10(q_sum)              # (m,)
                loss = model.pinball_loss(pred_log_aggr.unsqueeze(0), target.unsqueeze(0))
            else:
                pred = pred.squeeze(-1)
                delog_pred = torch.pow(10, pred)
                pred_sum = delog_pred.sum()
                pred_log_aggr = torch.log10(pred_sum)
                loss = l2_loss(pred_log_aggr, target).mean()
                if is_bnn:
                    loss = loss + args.klweight * model.kl_loss() / n_groups

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
        
def forward_full_grid(model, data, batch_size=65536):
    """Giống predict_in_batches_ensemble nhưng GIỮ LẠI đồ thị gradient (không
    dùng torch.no_grad()), để train() có thể backprop qua toàn bộ lưới.
    Trả về tensor CUDA (N, C) — C tuỳ loại model (1, k, 2/4, hoặc m)."""
    outs = []
    n_samples = data.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = data[start:end].cuda()
        outs.append(model(batch))
    return torch.cat(outs, dim=0)


def aggregate_channels_torch(values, region_mask):
    """values: (N, C) tensor CUDA (lưới m*n pixel, C kênh output).
    region_mask: (m, n) tensor CUDA nhãn vùng (int).
    Tổng hợp TỪNG kênh riêng biệt bằng aggragate_torch (vốn chỉ nhận input
    1 kênh dạng (m, n)), rồi ghép lại. Trả về (n_groups, C), đã bỏ nhãn '0'."""
    n_channels = values.shape[1]
    outs = []
    for c in range(n_channels):
        v2d = values[:, c].reshape(region_mask.shape)
        agg = aggragate_torch(v2d, region_mask)[1:].squeeze(-1)  # (n_groups,)
        outs.append(agg)
    return torch.stack(outs, dim=1)  # (n_groups, C)


def train():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    model_flag = timestamp + '_' + params_str
    log_root = f"log/{model_flag}"
    print(log_root)
    summaryWriter = SummaryWriter(log_root)
    loss_bar = tqdm(range(1, EPOCH+1))
    min_rmse = 1e7
    region_mask_cuda = torch.from_numpy(RegionMask[1]).cuda()
    for epoch in loss_bar:
        raw_grid = forward_full_grid(model, in_data_grid)  # (m*n, C), CUDA, requires_grad

        if is_ep:
            delog = torch.pow(10, raw_grid)                              # (N, k)
            pred_log_aggr = torch.log10(aggregate_channels_torch(delog, region_mask_cuda))  # (n_groups, k)
            per_group_losses = [
                wta_loss(pred_log_aggr[g], re_data_aggr_by_division[g], args.wtaeps)
                for g in range(pred_log_aggr.shape[0])
            ]
            loss = torch.stack(per_group_losses).mean()
        elif is_pdp:
            mu, sigma = raw_grid[:, :1], raw_grid[:, 1:2]
            mu_sum = aggregate_channels_torch(torch.pow(10, mu), region_mask_cuda).squeeze(-1)
            pred_log_aggr = torch.log10(mu_sum)                            # (n_groups,)
            sigma2_sum = aggregate_channels_torch(sigma ** 2, region_mask_cuda).squeeze(-1)
            counts = aggregate_channels_torch(torch.ones_like(sigma), region_mask_cuda).squeeze(-1)
            # Xấp xỉ: RMS của sigma trên các pixel trong group (xem ghi chú
            # trong train_per_tpu()) — không lan truyền phương sai chặt chẽ.
            sigma_log_aggr = torch.sqrt(sigma2_sum / counts.clamp(min=1)) + 1e-6
            loss = (0.5 * ((pred_log_aggr - re_data_aggr_by_division) / sigma_log_aggr) ** 2
                    + torch.log(sigma_log_aggr)).mean()
        elif is_npdp:
            delog_q = torch.pow(10, raw_grid)                              # (N, m)
            pred_log_aggr = torch.log10(aggregate_channels_torch(delog_q, region_mask_cuda))  # (n_groups, m)
            loss = model.pinball_loss(pred_log_aggr, re_data_aggr_by_division)
        else:
            pop_p_grid = raw_grid.squeeze(-1)                               # (N,)
            delog_pop_p_grid = torch.pow(10, pop_p_grid).reshape(m, n)
            pop_p_grid_aggr_TPU = torch.log10(aggragate_torch(
                delog_pop_p_grid, region_mask_cuda)[1:]).squeeze(-1)
            loss = l2_loss(pop_p_grid_aggr_TPU, re_data_aggr_by_division).mean()
            if is_bnn:
                loss = loss + args.klweight * model.kl_loss() / (m * n)

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
        pop_p = predict_point_in_batches(model, in_data_grid).reshape((m, n))
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

def enable_dropout(model):
    """Bật lại Dropout ở chế độ train trong khi phần còn lại của model ở eval,
    để lấy được các dự đoán stochastic (MC Dropout) khi suy luận."""
    model.eval()
    for m_ in model.modules():
        if isinstance(m_, nn.Dropout):
            m_.train()


def mapping_interval(n_samples=None, percentiles=(2.5, 97.5), unit=0):
    """Dự đoán khoảng bất định theo mẫu predict_grid_interval, xử lý TỪNG
    đơn vị hành chính (bounding-box window theo RegionMask[unit], tương tự
    ndimage.find_objects trên mastergrid ở bản QRF) thay vì forward toàn
    lưới cùng lúc:
    - MC Dropout / BNN: lặp n_samples lần forward trên mỗi đơn vị.
    - Ensemble Prediction (EP): chỉ 1 forward, k thành viên output chính
      là các mẫu (không cần vòng lặp sample).
    - Parametric Distributional Prediction (PDP): chỉ 1 forward ra tham số
      phân phối (mu, sigma, ...), sau đó sample analytically (torch
      distributions) — không cần forward NN lại.
    - Non-Parametric Distributional Prediction (NPDP): chỉ 1 forward ra m
      quantile, coi trực tiếp các quantile này như "mẫu" (giống cách xử lý
      k thành viên của EP).
    Bộ nhớ đỉnh tỉ lệ với kích thước đơn vị hành chính lớn nhất, không phải
    toàn ảnh.
    """
    n_samples = n_samples or args.nsamples

    with torch.no_grad():
        if is_ep or is_pdp or is_npdp:
            model.eval()  # 1 forward duy nhất, không cần lặp sample
        elif is_bnn:
            model.eval()  # BayesianLinear tự sample trọng số mỗi forward, kể cả ở eval mode
        else:
            enable_dropout(model)  # MC Dropout: cần bật lại Dropout thủ công

        mask_arr = RegionMask[unit]
        pop_total_arr = PopDensity[unit] * Area[unit]

        nodata = 0.0
        mean_map = np.full((m, n), nodata, dtype=np.float32)
        pct_maps = {p: np.full((m, n), nodata, dtype=np.float32) for p in percentiles}

        max_label = int(np.nanmax(mask_arr))
        objects = ndimage.find_objects(mask_arr.astype(np.int32), max_label=max_label)
        district_ids = [d for d in range(1, max_label + 1) if objects[d - 1] is not None]
        print(f'Processing {len(district_ids)} admin units')

        for district_id in tqdm(district_ids, desc='Uncertainty sampling per admin unit'):
            row_slice, col_slice = objects[district_id - 1]
            mask_win = mask_arr[row_slice, col_slice]
            district_mask = (mask_win == district_id)
            if not district_mask.any():
                continue

            rows = np.arange(row_slice.start, row_slice.stop)
            cols = np.arange(col_slice.start, col_slice.stop)
            row_idx, col_idx = np.meshgrid(rows, cols, indexing='ij')
            sel_rows = row_idx[district_mask]
            sel_cols = col_idx[district_mask]
            flat_idx = sel_rows * n + sel_cols

            batch_x = in_data_grid[torch.from_numpy(flat_idx).long()]

            if is_ep or is_npdp:
                # EP: 1 forward -> (n_valid, k), coi k thành viên như k "mẫu"
                # NPDP: 1 forward -> (n_valid, m), coi m quantile như "mẫu"
                raw = predict_in_batches_ensemble(model, batch_x)  # (n_valid, k hoặc m)
                samples = raw.T.astype(np.float32)                # (k hoặc m, n_valid)
            elif is_pdp:
                # 1 forward -> tham số phân phối, sau đó sample analytically
                raw = torch.from_numpy(predict_in_batches_ensemble(model, batch_x))
                if model.distribution == 'gaussian':
                    dist = torch.distributions.Normal(raw[:, 0], raw[:, 1])
                else:
                    dist = torch.distributions.StudentT(
                        df=raw[:, 2], loc=raw[:, 0], scale=raw[:, 3])
                samples = dist.sample((n_samples,)).numpy().astype(np.float32)  # (n_samples, n_valid)
            else:
                samples = np.empty((n_samples, flat_idx.size), dtype=np.float32)
                for s in range(n_samples):
                    samples[s] = predict_in_batches(model, batch_x)
            del batch_x
            gc.collect()

            samples = np.power(10, samples)  # de-log
            census_total = pop_total_arr[row_slice, col_slice][district_mask][0]
            sample_totals = samples.sum(axis=1)
            factors = np.divide(
                census_total, sample_totals,
                out=np.zeros_like(sample_totals, dtype=np.float64),
                where=sample_totals > 0
            )
            samples *= factors[:, None]

            mean_map[sel_rows, sel_cols] = samples.mean(axis=0)
            for p in percentiles:
                mean_map_tag = pct_maps[p]
                mean_map_tag[sel_rows, sel_cols] = np.percentile(samples, p, axis=0)

            del samples
            gc.collect()

        model.eval()  # tắt dropout trở lại sau khi lấy mẫu xong (no-op với BNN)

        population_redistributed_sum_by_su = group_aggregation_(
            mean_map, RegionMask[0], False, method='sum')[1][1:]
        np.savetxt(f'map/{args.ckp}_interval.csv',
                   population_redistributed_sum_by_su, delimiter=',')

        base = args.savepath
        stem, ext = os.path.splitext(base)
        copy_geoinfo_and_save_image(args.map, mean_map, f'{stem}_mean{ext}')
        for p in percentiles:
            tag = f'p{str(p).replace(".", "")}'
            copy_geoinfo_and_save_image(args.map, pct_maps[p], f'{stem}_{tag}{ext}')

        print('Saved mean +', list(percentiles), 'percentile rasters.')


def mapping():
    with torch.no_grad():
        pop_p = predict_point_in_batches(model, in_data_grid).reshape((m, n))

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
        pop_p = predict_point_in_batches(model, in_data_grid).reshape((m, n))
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
    elif args.mode == 'map_interval':
        mapping_interval()
    else:
        evaluation(model)