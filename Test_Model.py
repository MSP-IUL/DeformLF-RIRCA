import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from opt import opt
import h5py
import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim
from Main_Model import Net
from PIL import Image
import copy

# Ensure duplicate library error doesn't occur
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class DatasetFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromHdf5, self).__init__()
        
        hf = h5py.File(opt.data_path)
        self.LFI_ycbcr = hf.get('LFI_ycbcr') # [N,ah,aw,h,w,3]
        self.ang_out = opt.angular_out
        self.ang_in = opt.angular_in

    def __getitem__(self, index):
        H, W = self.LFI_ycbcr.shape[3:5]
        lfi_ycbcr = self.LFI_ycbcr[index]
        lfi_ycbcr = lfi_ycbcr[:opt.angular_out, :opt.angular_out, :].reshape(-1, H, W, 3)
        ind_all = np.arange(self.ang_out*self.ang_out).reshape(self.ang_out, self.ang_out)
        delt = (self.ang_out-1) // (self.ang_in-1)
        ind_source = ind_all[0:self.ang_out:delt, 0:self.ang_out:delt]
        ind_source = ind_source.reshape(-1)
        input = lfi_ycbcr[ind_source, :, :, 0]

        ### SAI2Lens ###
        NumView = 2
        a = 0
        LF = np.zeros((1, H * NumView, W * NumView))
        for i in range(NumView):
            for j in range(NumView):
                img = input[a, :, :]
                img = img[np.newaxis, :, :]
                LF[:, i::NumView, j::NumView] = img  # 1x1024x1024
                a = a + 1
        lenslet_data = LF
        allah = self.ang_in
        allaw = self.ang_in
        LFI = np.zeros((1, H, W, allah, allaw))
        for v in range(allah):
            for u in range(allah):
                sub = lenslet_data[:, v::allah, u::allah]
                LFI[:, :, :, v, u] = sub[:, 0:H, 0:W]
        LFI = LFI.reshape(H, W, 2, 2)
        ### target
        target_y = lfi_ycbcr[:, :, :, 0]
        input = torch.from_numpy(input.astype(np.float32)/255.0)
        target_y = torch.from_numpy(target_y.astype(np.float32)/255.0)
        LFI = torch.from_numpy(LFI.astype(np.float32)/255.0)
        # keep cbcr for RGB reconstruction (Using Ground truth just for visual results)
        lfi_ycbcr = torch.from_numpy(lfi_ycbcr.astype(np.float32)/255.0) 
        
        return ind_source, input, target_y, lfi_ycbcr, LFI
        
    def __len__(self):
        return self.LFI_ycbcr.shape[0]


# Load test dataset
print('===> Loading test datasets')
test_set = DatasetFromHdf5(opt)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
print(f'Loaded {len(test_loader)} LFIs from {opt.data_path}')

# Build model
print("Building model")
model_view = Net(opt).to(device)  # Move the model to GPU

# Load model weights
print(f"Loading model weights from {opt.model_path}")
checkpoint = torch.load(opt.model_path, map_location=device)
model_view.load_state_dict(checkpoint['model'])
print(f"Loaded model: {opt.model_path}")

# Evaluation mode
model_view.eval()

# Helper functions for PSNR, saving images, and SSIM
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def ycbcr2rgb(ycbcr):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:, 0] -= 16. / 255.
    rgb[:, 1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)

def save_img(pred_y, pred_ycbcr, lfi_no):
    if opt.save_img:
        save_dir = f'saveImg/HCI_new_test{opt.test_dataset}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(opt.angular_out * opt.angular_out):
            img_ycbcr = pred_ycbcr[0, i]
            img_ycbcr[:, :, 0] = pred_y[0, i]
            img_rgb = ycbcr2rgb(img_ycbcr)
            img = (img_rgb.clip(0, 1) * 255.0).astype(np.uint8)
            img_name = f'{save_dir}/SynLFI{lfi_no}_view{i}.png'
            Image.fromarray(img).convert('RGB').save(img_name)

def compute_quan(pred_y, target_y, ind_source, csv_name, lfi_no):
    view_list, view_psnr_y, view_ssim_y = [], [], []
    for i in range(opt.angular_out * opt.angular_out):
        if i not in ind_source:
            cur_target_y = target_y[0, i]
            cur_pred_y = pred_y[0, i]
            cur_psnr_y = compute_psnr(cur_target_y, cur_pred_y)
            cur_ssim_y = compare_ssim((cur_target_y * 255.0).astype(np.uint8),
                                      (cur_pred_y * 255.0).astype(np.uint8),
                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            view_list.append(i)
            view_psnr_y.append(cur_psnr_y)
            view_ssim_y.append(cur_ssim_y)
    dataframe_lfi = pd.DataFrame(
        {'targetView_LFI{}'.format(lfi_no): view_list, 'psnr Y': view_psnr_y, 'ssim Y': view_ssim_y})
    dataframe_lfi.to_csv(csv_name, index=False, sep=',', mode='a')
    return np.mean(view_psnr_y), np.mean(view_ssim_y)

# Test function
def test():
    csv_name = f'quan_results/HCI_new_test{opt.test_dataset}.csv'
    if not os.path.exists('quan_results'):
        os.makedirs('quan_results')

    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            print(f'Testing LF {k}')
            # Move data to GPU
            ind_source, input, target_y, lfi_ycbcr, LFI = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].numpy(),
                batch[3].numpy(),
                batch[4].to(device),
            )

            # Perform inference
            pred_y = model_view(ind_source, input, LFI)
            pred_y = pred_y.cpu().numpy()

            # Save synthesized images
            save_img(pred_y, lfi_ycbcr, k)

            # Compute and save metrics
            bd = 22
            pred_y = pred_y[:, :, bd:-bd, bd:-bd]
            target_y = target_y[:, :, bd:-bd, bd:-bd]
            lf_psnr, lf_ssim = compute_quan(pred_y, target_y, ind_source, csv_name, k)
            print(f'LF {k} PSNR: {lf_psnr:.2f}, SSIM: {lf_ssim:.3f}')

# Run testing
test()
