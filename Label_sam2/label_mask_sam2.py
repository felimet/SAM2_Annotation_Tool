# %% [markdown]
# ## 設定

# %%
import torch
from PIL import Image
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


# %% [markdown]
# ## 函數

# %%
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# %%
def save_id_mask(out_mask_logits, ann_frame_dir, ann_frame_idx, show_visualization=False):
    """
    將遮罩轉換為物件ID圖像並儲存
    
    參數:
    out_mask_logits: 模型輸出的遮罩邏輯值
    ann_frame_dir: 目錄路徑
    ann_frame_idx: 當前處理的幀索引
    show_visualization: 是否顯示視覺化結果 (預設: False)
    
    返回:
    id_mask: 生成的ID遮罩陣列
    mask_path: 儲存的遮罩檔案路徑
    """
    
    # 將輸出轉為布林遮罩並轉換為numpy陣列
    masks_np = [(mask > 0.0).cpu().numpy() for mask in out_mask_logits]
    
    # 獲取遮罩的高度和寬度（參考 show_mask 函數的處理方式）
    h, w = masks_np[0].shape[-2:]
    # print(f"遮罩形狀: 高={h}, 寬={w}")
    
    # 創建一個空的ID遮罩，背景為0
    id_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 用物件ID值填充每個遮罩對應的區域
    for mask_idx, mask in enumerate(masks_np):
        # 確保遮罩形狀正確，參考 show_mask 函數中的 reshape 方式
        binary_mask = mask.reshape(h, w)
        
        # 將二值遮罩區域設定為物件ID值 (mask_idx + 1 避免與背景值0混淆)
        object_id = mask_idx + 1
        id_mask[binary_mask] = object_id
    
    # 確保儲存目錄存在
    save_dir = os.path.join(ann_frame_dir, "id_masks")
    os.makedirs(save_dir, exist_ok=True)
    
    # 儲存物件ID遮罩
    mask_image = Image.fromarray(id_mask)
    mask_path = os.path.join(save_dir, f"{ann_frame_idx:06d}.png")
    mask_image.save(mask_path)
    # print(f"已儲存ID遮罩至 {mask_path}")
    # print(f"遮罩中的物件ID值: {np.unique(id_mask)}")
    
    # 可視化ID遮罩 (如果需要)
    if show_visualization:
        plt.figure(figsize=(9, 6))
        plt.title(f"Object ID Mask (frame {ann_frame_idx})")
        plt.imshow(id_mask, cmap='jet')  
        plt.colorbar(label='Object ID')
        plt.show()
    
    return id_mask, mask_path


# %%
%cd D:/Anycode/label-sam2


# %%
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)


# %%
video_dir = "D:/AnyCode/dataset/videos_frame/20250313_170517-170544_clip_00-08_to_00-15"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)


# %%
# 如果您使用此“ inference_state”運行任何以前的跟踪，請首先通過`reset_state`重置它。
predictor.reset_state(inference_state)


# %%
import matplotlib.pyplot as plt
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))


# %% [markdown]
# ## 標記

# %%
prompts = {}  # hold all the clicks we add for visualization


# %%
ann_frame_idx = 170  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (200, 300) to get started on the first object
points = np.array([[1461, 372], [1484, 493], [1451, 551], [1596, 492], [1463, 500]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0, 1, 1, 1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(16, 9))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)


# %% [markdown]
# ## 預測
# 包含反向預測、儲存物件ID mask

# %%
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
id_mask_path = os.path.join('D:/AnyCode/dataset', 'video_annotations')
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    id_mask, mask_path = save_id_mask(
            out_mask_logits=out_mask_logits,
            ann_frame_dir=id_mask_path,
            ann_frame_idx=out_frame_idx,
            show_visualization=False
        )


# %% [markdown]
# ## 結果呈現

# %%
# render the segmentation results every few frames
# vis_frame_stride = 30
# plt.close("all")
# for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
#     plt.figure(figsize=(6, 4))
#     plt.title(f"frame {out_frame_idx}")
#     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
#     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


# %%
# id_mask_path = os.path.join('D:/AnyCode/dataset', 'video_annotations')
# id_mask, mask_path = save_id_mask(
#     out_mask_logits=out_mask_logits,
#     ann_frame_dir=id_mask_path,
#     ann_frame_idx=ann_frame_idx,
#     show_visualization=False  # 設為True可視覺化結果
# )



