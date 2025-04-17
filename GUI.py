import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading  # 用於在背景執行耗時任務
import time  # 用於計時影像載入過程

# 用於處理資源路徑，確保開發和打包環境下均可正確運行
def resource_path(relative_path):
    """獲取資源的絕對路徑，適用於開發和 PyInstaller 打包環境"""
    try:
        # PyInstaller 創建臨時文件夾並將路徑存儲在 _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


# --- 啟動畫面 ---
def show_splash_screen():
    """顯示啟動畫面並自動關閉"""
    splash = tk.Toplevel()
    splash.overrideredirect(True)  # 無邊框窗口
    
    # 獲取屏幕尺寸以便置中顯示
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    
    # 載入啟動圖像
    img_path = resource_path(os.path.join("favicon_io", "android-chrome-512x512.png"))
    try:
        img = Image.open(img_path)
        img = img.resize((300, 300))  # 調整到合適大小
        splash_img = ImageTk.PhotoImage(img)
        
        # 設置窗口大小
        width, height = 400, 400
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        splash.geometry(f"{width}x{height}+{x}+{y}")
        
        # 設置背景色
        splash.configure(background='#1E3A8A')
        
        # 添加標誌圖像
        img_label = tk.Label(splash, image=splash_img, bg='#1E3A8A')
        img_label.place(relx=0.5, rely=0.45, anchor='center')
        
        # 添加標題
        title_label = tk.Label(splash, text="SAM2 標註工具", font=("Arial", 18, "bold"), 
                              fg='white', bg='#1E3A8A')
        title_label.place(relx=0.5, rely=0.75, anchor='center')
        
        # 添加載入中文字
        loading_label = tk.Label(splash, text="正在載入，請稍候...", font=("Arial", 10), 
                               fg='white', bg='#1E3A8A')
        loading_label.place(relx=0.5, rely=0.85, anchor='center')
        
        # 更新界面
        splash.update()
        
        # 保持圖像引用
        splash.img = splash_img
        
        # 設置在主窗口顯示後關閉啟動畫面
        splash.after(3000, splash.destroy)
        
        return splash
    except Exception as e:
        print(f"啟動畫面加載失敗: {e}")
        if splash.winfo_exists():
            splash.destroy()
        return None


# --- 從原始腳本導入或重新定義必要的函數和設定 ---

# 檢查並設定計算裝置
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available(): # MPS 支援可能不穩定，暫時禁用
    #     device = torch.device("mps")
    #     print(
    #         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
    #         "give numerically different outputs and sometimes degraded performance on MPS. "
    #         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    #     )
    else:
        device = torch.device("cpu")
    print(f"使用裝置: {device}")

    if device.type == "cuda":
        # 為 Ampere GPU 開啟 tfloat32
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
except Exception as e:
    print(f"設定 Pytorch 裝置時發生錯誤: {e}")
    device = torch.device("cpu")
    print(f"已切換回 CPU: {device}")


# --- 視覺化函數 (調整以適應 Tkinter) ---
def show_mask_on_ax(mask, ax, obj_id=None, random_color=False):
    """在 Matplotlib Axes 上顯示遮罩"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id % 10  # 使用模數以循環使用顏色
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points_on_ax(coords, labels, ax, marker_size=50):
    """在 Matplotlib Axes 上顯示點"""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.0)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.0)

# --- 遮罩儲存函數 ---


def save_id_mask(out_mask_logits, ann_frame_dir, ann_frame_idx, show_visualization=False):
    """
    將遮罩轉換為物件ID影像並儲存

    參數:
    out_mask_logits: 模型輸出的遮罩邏輯值 (Tensor list)
    ann_frame_dir: 目錄路徑
    ann_frame_idx: 當前處理的畫格索引
    show_visualization: 是否顯示視覺化結果 (在此 GUI 中通常為 False)

    返回:
    id_mask: 生成的ID遮罩陣列
    mask_path: 儲存的遮罩檔案路徑
    """
    try:
        # 將輸出轉為布林遮罩並轉換為numpy陣列
        # 確保在 CPU 上操作 NumPy 陣列
        masks_np = [(mask > 0.0).cpu().numpy() for mask in out_mask_logits]

        if not masks_np:
            print(f"警告: 畫格 {ann_frame_idx} 沒有有效的遮罩輸出。")
            return None, None

        # 獲取遮罩的高度和寬度
        h, w = masks_np[0].shape[-2:]

        # 創建一個空的ID遮罩，背景為0
        id_mask = np.zeros((h, w), dtype=np.uint8)

        # 用物件ID值填充每個遮罩對應的區域
        for mask_idx, mask in enumerate(masks_np):
            binary_mask = mask.reshape(h, w)
            object_id = mask_idx + 1  # 物件 ID 從 1 開始
            id_mask[binary_mask] = object_id

        # 確保儲存目錄存在
        save_dir = os.path.join(ann_frame_dir, "id_masks")
        os.makedirs(save_dir, exist_ok=True)

        # 儲存物件ID遮罩，檔案命名從 1 開始，而不是從 0 開始
        mask_image = Image.fromarray(id_mask)
        # 將 frame_idx + 1 使命名從 1 開始
        mask_path = os.path.join(save_dir, f"{(ann_frame_idx + 1):06d}.png")
        mask_image.save(mask_path)
        # print(f"已儲存ID遮罩至 {mask_path}")

        return id_mask, mask_path
    except Exception as e:
        print(f"儲存 ID 遮罩時發生錯誤 (畫格 {ann_frame_idx}): {e}")
        return None, None


# --- SAM 2 模型相關導入 ---
try:
    # 假設 sam2 套件在環境中可用
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    messagebox.showerror("錯誤", "找不到 'sam2' 套件。請確保已正確安裝 SAM 2。")
    exit()


# --- 主應用程式類別 ---
class SAM2_GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SAM 2 互動式標註工具")
        self.geometry("1200x800")  # 設定初始視窗大小

        # --- 變數 ---
        self.model_cfg = tk.StringVar(
            value=resource_path(os.path.join("sam2", "sam2_hiera_l.yaml")))  # 預設模型配置
        self.checkpoint_path = tk.StringVar(
            value=resource_path(os.path.join("checkpoints", "sam2.1_hiera_large.pt")))  # 預設權重檔
        self.video_dir = tk.StringVar()
        self.output_dir = tk.StringVar(
            value="./video_annotations")  # 預設輸出目錄
        self.frame_names = []
        self.current_frame_index = tk.IntVar(value=0)
        self.predictor = None
        self.inference_state = None
        self.current_image = None  # PIL Image
        self.current_photo_image = None  # Tkinter PhotoImage
        self.points = []  # 儲存點座標 (x, y)
        self.labels = []  # 儲存點標籤 (1: 正, 0: 負)
        self.current_object_id = 1  # 當前正在標註的物件 ID
        self.prompts = {}  # 儲存每個物件的提示 {obj_id: (points_np, labels_np)}
        self.current_masks = {}  # 儲存目前畫格顯示的遮罩 {obj_id: mask_np}
        self.is_processing = False  # 標記是否正在處理耗時任務
        self.auto_apply = tk.BooleanVar(value=True)  # 控制是否自動套用提示點

        # --- GUI 佈局 ---
        self.create_widgets()

        # --- Matplotlib 圖形 ---
        self.fig, self.ax = plt.subplots(figsize=(10, 6))  # 調整圖形大小
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout()  # 自動調整佈局

        # --- 事件綁定 ---
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.current_frame_index.trace_add(
            "write", self.update_frame_display)  # 當索引改變時更新畫面

        # --- 顯示操作說明 ---
        self.show_welcome_message()

    def create_widgets(self):
        # --- 主框架 ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 左側控制面板 ---
        control_panel = ttk.Frame(main_frame, width=300)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # --- 模型設定 ---
        model_frame = ttk.LabelFrame(control_panel, text="模型設定")
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="設定檔:").grid(
            row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(model_frame, textvariable=self.model_cfg,
                  width=30).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(model_frame, text="瀏覽", command=lambda: self.browse_file(
            self.model_cfg, "選擇模型設定檔", [("YAML files", "*.yaml")])).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(model_frame, text="權重檔:").grid(
            row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(model_frame, textvariable=self.checkpoint_path,
                  width=30).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(model_frame, text="瀏覽", command=lambda: self.browse_file(
            self.checkpoint_path, "選擇模型權重檔", [("PyTorch files", "*.pt")])).grid(row=1, column=2, padx=5, pady=2)

        self.load_model_button = ttk.Button(
            model_frame, text="載入模型", command=self.load_model)
        self.load_model_button.grid(row=2, column=0, columnspan=3, pady=5)

        # --- 影片與輸出 ---
        io_frame = ttk.LabelFrame(control_panel, text="影片與輸出")
        io_frame.pack(fill=tk.X, pady=5)

        ttk.Label(io_frame, text="影片畫格目錄:").grid(
            row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(io_frame, textvariable=self.video_dir, width=30).grid(
            row=0, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="瀏覽", command=self.browse_video_dir).grid(
            row=0, column=2, padx=5, pady=2)

        ttk.Label(io_frame, text="輸出目錄:").grid(
            row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(io_frame, textvariable=self.output_dir,
                  width=30).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="瀏覽", command=lambda: self.browse_directory(
            self.output_dir, "選擇輸出目錄")).grid(row=1, column=2, padx=5, pady=2)

        # --- 畫格導覽 ---
        nav_frame = ttk.LabelFrame(control_panel, text="畫格導覽")
        nav_frame.pack(fill=tk.X, pady=5)

        # 新增畫格拉桿和輸入框
        slider_frame = ttk.Frame(nav_frame)
        slider_frame.pack(fill=tk.X, pady=5)

        self.prev_button = ttk.Button(
            slider_frame, text="◀", width=2, command=self.prev_frame, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=2)

        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=0, orient="horizontal",
                                      variable=self.current_frame_index, command=self.on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.next_button = ttk.Button(
            slider_frame, text="▶", width=2, command=self.next_frame, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=2)

        # 畫格數字顯示與輸入
        frame_num_frame = ttk.Frame(nav_frame)
        frame_num_frame.pack(fill=tk.X, pady=3)

        self.frame_label = ttk.Label(frame_num_frame, text="畫格:")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        vcmd = (self.register(self.validate_frame_entry), '%P')
        self.frame_entry = ttk.Entry(
            frame_num_frame, width=6, validate="key", validatecommand=vcmd)
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        self.frame_entry.bind("<Return>", self.on_frame_entry)

        self.total_frames_label = ttk.Label(frame_num_frame, text="/ 0")
        self.total_frames_label.pack(side=tk.LEFT)

        # 跳轉按鈕
        self.goto_button = ttk.Button(
            frame_num_frame, text="跳轉", command=self.goto_frame, width=6)
        self.goto_button.pack(side=tk.LEFT, padx=5)

        # --- 滑鼠操作說明 ---
        mouse_frame = ttk.LabelFrame(control_panel, text="滑鼠操作說明")
        mouse_frame.pack(fill=tk.X, pady=5)

        mouse_info = ttk.Frame(mouse_frame)
        mouse_info.pack(fill=tk.X, pady=5, padx=5)

        # 左鍵說明
        left_frame = ttk.Frame(mouse_info)
        left_frame.pack(fill=tk.X, pady=2)
        left_icon = ttk.Label(left_frame, text="●", foreground="green")
        left_icon.pack(side=tk.LEFT, padx=(0, 5))
        left_desc = ttk.Label(left_frame, text="左鍵點擊: 添加正向標註點（目標物件）")
        left_desc.pack(side=tk.LEFT)

        # 右鍵說明
        right_frame = ttk.Frame(mouse_info)
        right_frame.pack(fill=tk.X, pady=2)
        right_icon = ttk.Label(right_frame, text="●", foreground="red")
        right_icon.pack(side=tk.LEFT, padx=(0, 5))
        right_desc = ttk.Label(right_frame, text="右鍵點擊: 添加負向標註點（背景/修飾）")
        right_desc.pack(side=tk.LEFT)

        # 自動套用選項
        auto_apply_frame = ttk.Frame(mouse_info)
        auto_apply_frame.pack(fill=tk.X, pady=5)
        self.auto_apply_check = ttk.Checkbutton(auto_apply_frame, text="點擊時自動套用提示點",
                                                variable=self.auto_apply)
        self.auto_apply_check.pack(side=tk.LEFT)

        # --- 標註控制 ---
        annotate_frame = ttk.LabelFrame(control_panel, text="標註控制")
        annotate_frame.pack(fill=tk.X, pady=5)

        obj_id_frame = ttk.Frame(annotate_frame)
        obj_id_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(obj_id_frame, text="目前物件 ID:").pack(side=tk.LEFT)
        self.obj_id_spinbox = ttk.Spinbox(
            obj_id_frame, from_=1, to=100, width=5, command=self.update_object_id)
        self.obj_id_spinbox.pack(side=tk.LEFT, padx=5)
        self.obj_id_spinbox.set(self.current_object_id)  # 設定初始值

        # 即使保留「套用提示點」按鈕，但將其作為備用選項
        button_frame = ttk.Frame(annotate_frame)
        button_frame.pack(fill=tk.X, pady=3)

        self.add_points_button = ttk.Button(
            button_frame, text="手動套用提示點", command=self.apply_prompts, state=tk.DISABLED)
        self.add_points_button.pack(
            side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)

        self.clear_points_button = ttk.Button(
            button_frame, text="清除目前點", command=self.clear_current_points, state=tk.DISABLED)
        self.clear_points_button.pack(
            side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)

        self.clear_object_button = ttk.Button(
            annotate_frame, text="清除此物件標註", command=self.clear_object_annotation, state=tk.DISABLED)
        self.clear_object_button.pack(pady=5, padx=5, fill=tk.X)

        # --- 處理 ---
        process_frame = ttk.LabelFrame(control_panel, text="處理")
        process_frame.pack(fill=tk.X, pady=5)

        self.propagate_button = ttk.Button(
            process_frame, text="執行影片推理", command=self.run_propagation, state=tk.DISABLED)
        self.propagate_button.pack(pady=5, padx=5, fill=tk.X)

        # --- 狀態欄 ---
        self.status_bar = ttk.Label(
            control_panel, text="狀態: 未載入模型", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(10, 0), side=tk.BOTTOM)

        # --- 右側影像顯示區 ---
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def show_welcome_message(self):
        """顯示歡迎訊息與操作指南"""
        welcome_msg = (
            "歡迎使用 SAM 2 互動式標註工具！\n\n"
            "開始使用步驟：\n"
            "1. 載入模型（預設路徑已設定）\n"
            "2. 選擇影片畫格目錄\n"
            "3. 在影像上使用左鍵（添加目標）或右鍵（添加背景）標註\n"
            "4. 標註完成後執行影片推理\n\n"
            "提示：點擊時會自動套用標註，無需額外按「套用提示點」。"
        )
        messagebox.showinfo("使用指南", welcome_msg)

    # --- 瀏覽函數 ---
    def browse_file(self, var, title, filetypes):
        """通用檔案瀏覽函數"""
        filename = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if filename:
            var.set(filename)

    def browse_directory(self, var, title):
        """通用目錄瀏覽函數"""
        dirname = filedialog.askdirectory(title=title)
        if dirname:
            var.set(dirname)

    def browse_video_dir(self):
        """瀏覽影片畫格目錄並載入畫格"""
        dirname = filedialog.askdirectory(title="選擇影片畫格目錄")
        if dirname:
            self.video_dir.set(dirname)
            self.load_frames()

    def validate_frame_entry(self, new_value):
        """驗證畫格輸入框的值是否有效"""
        if new_value == "":
            return True
        try:
            value = int(new_value)
            if value >= 0:
                return True
            return False
        except ValueError:
            return False

    def on_frame_entry(self, event):
        """處理畫格輸入框的Enter鍵事件"""
        self.goto_frame()

    def goto_frame(self):
        """跳轉到指定畫格"""
        try:
            idx = int(self.frame_entry.get())
            if 0 <= idx < len(self.frame_names):
                self.current_frame_index.set(idx)
                # 清除當前畫格的臨時點和遮罩
                self.points = []
                self.labels = []
                self.current_masks = {}
            else:
                messagebox.showwarning(
                    "警告", f"畫格索引必須在 0 到 {len(self.frame_names)-1} 之間")
                self.frame_entry.delete(0, tk.END)
                self.frame_entry.insert(0, str(self.current_frame_index.get()))
        except ValueError:
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(self.current_frame_index.get()))

    def on_slider_change(self, value):
        """滑動條值變化時的回調"""
        try:
            # 將浮點數值轉為整數
            idx = int(float(value))
            # 更新輸入框的值
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(idx))
            # 清除當前畫格的臨時點和遮罩（如果變化了）
            if idx != self.current_frame_index.get():
                self.points = []
                self.labels = []
                self.current_masks = {}
        except ValueError:
            pass

    # --- 模型與影片載入 ---
    def load_model(self):
        """載入 SAM 2 模型"""
        if self.is_processing:
            messagebox.showwarning("警告", "正在處理中，請稍候。")
            return

        cfg_path = self.model_cfg.get()
        chk_path = self.checkpoint_path.get()

        if not os.path.exists(cfg_path):
            messagebox.showerror("錯誤", f"找不到模型設定檔: {cfg_path}")
            return
        if not os.path.exists(chk_path):
            messagebox.showerror("錯誤", f"找不到模型權重檔: {chk_path}")
            return

        self.update_status("正在載入模型...")
        self.set_ui_state(tk.DISABLED)  # 禁用介面

        def _load():
            try:
                # 使用 bfloat16 (如果 CUDA 可用)
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        self.predictor = build_sam2_video_predictor(
                            cfg_path, chk_path, device=device)
                else:
                    self.predictor = build_sam2_video_predictor(
                        cfg_path, chk_path, device=device)

                self.after(0, self.on_model_loaded)  # 在主線程中更新 UI
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "模型載入錯誤", f"載入模型時發生錯誤: {e}"))
                self.after(0, self.reset_ui_after_error)

        # 在背景線程中載入模型
        threading.Thread(target=_load, daemon=True).start()

    def on_model_loaded(self):
        """模型成功載入後的回調函數"""
        self.update_status("模型載入成功。請選擇影片目錄。")
        self.set_ui_state(tk.NORMAL)  # 恢復介面
        self.load_model_button.config(state=tk.DISABLED)  # 禁用載入按鈕
        # 如果影片目錄已選，則初始化 predictor 狀態
        if self.video_dir.get() and self.frame_names:
            self.initialize_predictor_state()
        else:
            # 啟用標註相關按鈕，但只有在畫格載入後才真正可用
            self.enable_annotation_buttons(False)

    def reset_ui_after_error(self):
        """發生錯誤時重置 UI"""
        self.update_status("錯誤，請重試。")
        self.set_ui_state(tk.NORMAL)  # 恢復介面以便重試
        self.is_processing = False

    def load_frames(self):
        """從選定的目錄載入影片畫格並顯示進度條"""
        video_path = self.video_dir.get()
        if not video_path or not os.path.isdir(video_path):
            messagebox.showerror("錯誤", "請先選擇有效的影片畫格目錄。")
            return

        try:
            # 獲取目錄中的所有可能的影像檔案
            image_files = [p for p in os.listdir(video_path)
                           if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]

            if not image_files:
                messagebox.showerror("錯誤", f"在 '{video_path}' 中找不到有效的圖像畫格。")
                self.video_dir.set("")
                return

            # 建立並顯示進度對話方塊
            progress_window = tk.Toplevel(self)
            progress_window.title("載入畫格中")
            progress_window.geometry("400x120")
            progress_window.resizable(False, False)
            progress_window.transient(self)  # 設為主視窗的子視窗
            progress_window.grab_set()  # 阻止與其他視窗互動

            # 置中顯示
            progress_window.geometry("+%d+%d" % (
                self.winfo_rootx() + (self.winfo_width() - 400) / 2,
                self.winfo_rooty() + (self.winfo_height() - 120) / 2))

            # 進度條標籤
            format_label = os.path.splitext(
                image_files[0])[-1].upper()[1:]  # 取得格式（如JPEG, PNG）
            progress_text = tk.StringVar(value=f"畫格載入中 ({format_label}): 0%")
            ttk.Label(progress_window, textvariable=progress_text).pack(
                pady=(15, 5))

            # 進度條
            progress_bar = ttk.Progressbar(
                progress_window, orient="horizontal", length=350, mode="determinate")
            progress_bar.pack(pady=5, padx=25)

            # 詳細進度資訊
            details_var = tk.StringVar(
                value="0/{} [00:00<00:00, 0.00張/秒]".format(len(image_files)))
            details_label = ttk.Label(
                progress_window, textvariable=details_var)
            details_label.pack(pady=5)

            # 取消按鈕（可選）
            cancel_button = ttk.Button(progress_window, text="取消", command=lambda: setattr(
                self, '_cancel_loading', True))
            cancel_button.pack(pady=5)

            # 初始化載入狀態
            self._cancel_loading = False
            total_files = len(image_files)
            loaded_files = 0
            start_time = time.time()
            self.frame_names = []

            def update_progress():
                """更新進度顯示"""
                nonlocal loaded_files, start_time

                if self._cancel_loading:
                    progress_window.destroy()
                    self.update_status("畫格載入已取消。")
                    return

                if loaded_files >= total_files:
                    # 完成載入
                    elapsed = time.time() - start_time
                    rate = total_files / elapsed if elapsed > 0 else 0

                    progress_text.set(f"畫格載入完成 ({format_label}): 100%")
                    progress_bar["value"] = 100
                    details_var.set(
                        f"{total_files}/{total_files} [時間:{elapsed:.2f}秒, 平均:{rate:.2f}張/秒]")

                    # 短暫顯示完成狀態後關閉視窗
                    self.after(1000, progress_window.destroy)

                    # 設定完成後的畫格顯示
                    self.finalize_frame_loading(total_files)
                    return

                # 計算進度和預估剩餘時間
                percent = int((loaded_files / total_files) * 100)
                elapsed = time.time() - start_time
                rate = loaded_files / elapsed if elapsed > 0 else 0
                remaining = (total_files - loaded_files) / \
                    rate if rate > 0 else 0

                # 更新進度顯示
                progress_text.set(f"畫格載入中 ({format_label}): {percent}%")
                progress_bar["value"] = percent

                # 格式化時間顯示
                elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
                remaining_str = time.strftime("%M:%S", time.gmtime(remaining))
                details_var.set(
                    f"{loaded_files}/{total_files} [{elapsed_str}<{remaining_str}, {rate:.2f}張/秒]")

                # 排程下一批畫格載入
                self.after(10, load_batch)

            def load_batch():
                """載入一批畫格"""
                nonlocal loaded_files

                if self._cancel_loading:
                    update_progress()
                    return

                # 每批載入的畫格數量
                batch_size = 10
                end_idx = min(loaded_files + batch_size, total_files)

                try:
                    for i in range(loaded_files, end_idx):
                        # 確認檔案可以正確排序
                        filename = image_files[i]
                        try:
                            # 嘗試將檔名轉換為數字（不包含副檔名）
                            int(os.path.splitext(filename)[0])
                            # 驗證可以打開圖像
                            image_path = os.path.join(video_path, filename)
                            with Image.open(image_path) as img:
                                img.verify()  # 驗證圖像檔案完整性
                            self.frame_names.append(filename)
                        except (ValueError, IOError, Image.DecompressionBombError) as e:
                            print(f"跳過無效圖像: {filename}, 錯誤: {e}")
                            continue

                    loaded_files = end_idx
                except Exception as e:
                    messagebox.showerror("載入錯誤", f"載入畫格時發生錯誤: {e}")
                    progress_window.destroy()
                    return

                update_progress()

            # 開始載入第一批
            self.after(50, load_batch)

        except Exception as e:
            messagebox.showerror("載入畫格錯誤", f"載入畫格時發生錯誤: {e}")
            self.frame_names = []
            self.video_dir.set("")

    def finalize_frame_loading(self, total_frames):
        """完成畫格載入後的設置"""
        # 根據檔名排序畫格
        try:
            self.frame_names = sorted(
                self.frame_names, key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            messagebox.showwarning("警告", "部分畫格檔名無法正確排序，可能會影響顯示順序。")

        # 重置到第一畫格
        self.current_frame_index.set(0)

        # 更新滑動條的範圍
        self.frame_slider.configure(from_=0, to=total_frames-1)

        # 更新畫格計數標籤
        self.total_frames_label.config(text=f"/ {total_frames-1}")

        # 設置畫格輸入框的值
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, "0")

        self.update_status(f"已載入 {total_frames} 畫格。")
        self.update_frame_display()  # 顯示第一畫格
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(
            state=tk.NORMAL if total_frames > 1 else tk.DISABLED)
        self.goto_button.config(
            state=tk.NORMAL if total_frames > 1 else tk.DISABLED)

        # 如果模型已載入，初始化 predictor 狀態
        if self.predictor:
            self.initialize_predictor_state()
        else:
            # 即使模型未載入，也啟用畫格導覽
            self.enable_navigation_buttons(True)

        # 提示載入完成
        messagebox.showinfo("載入完成", f"已成功載入 {total_frames} 個畫格！")

    def initialize_predictor_state(self):
        """初始化或重置 Predictor 狀態並顯示進度條"""
        if not self.predictor or not self.video_dir.get() or not self.frame_names:
            return

        self.update_status("正在初始化追蹤狀態...")
        self.set_ui_state(tk.DISABLED)

        # 建立並顯示進度對話方塊
        progress_window = tk.Toplevel(self)
        progress_window.title("初始化模型狀態")
        progress_window.geometry("400x150")
        progress_window.resizable(False, False)
        progress_window.transient(self)  # 設為主視窗的子視窗
        progress_window.grab_set()  # 阻止與其他視窗互動

        # 置中顯示
        progress_window.geometry("+%d+%d" % (
            self.winfo_rootx() + (self.winfo_width() - 400) / 2,
            self.winfo_rooty() + (self.winfo_height() - 150) / 2))

        # 階段說明標籤
        progress_text = tk.StringVar(value="正在初始化模型追蹤狀態...")
        main_label = ttk.Label(
            progress_window, textvariable=progress_text, font=("", 10, "bold"))
        main_label.pack(pady=(20, 10))

        # 進度指示區域
        progress_frame = ttk.Frame(progress_window)
        progress_frame.pack(fill=tk.X, pady=5, padx=20)

        # 處理階段標籤
        stage_text = tk.StringVar(value="載入影片資料")
        stage_label = ttk.Label(progress_frame, textvariable=stage_text)
        stage_label.pack(anchor=tk.W)

        # 進度條 (不確定模式)
        progress_bar = ttk.Progressbar(progress_frame, orient="horizontal",
                                    length=360, mode="indeterminate")
        progress_bar.pack(fill=tk.X, pady=5)
        progress_bar.start(15)  # 開始動畫

        # 狀態訊息
        status_var = tk.StringVar(value="處理中，請稍候...")
        status_label = ttk.Label(
            progress_frame, textvariable=status_var, font=("", 9, "italic"))
        status_label.pack(anchor=tk.W, pady=5)

        # 計時顯示
        timer_var = tk.StringVar(value="經過時間: 0:00")
        timer_label = ttk.Label(progress_window, textvariable=timer_var)
        timer_label.pack(side=tk.BOTTOM, anchor=tk.E, padx=20, pady=10)

        # 初始化計時器
        start_time = time.time()

        # 更新計時器函數
        def update_timer():
            if not progress_window.winfo_exists():
                return
            elapsed = int(time.time() - start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            timer_var.set(f"經過時間: {mins}:{secs:02d}")
            progress_window.after(1000, update_timer)

        # 啟動計時器
        update_timer()

        # 更新處理階段函數
        def update_stage(stage_name, status_message="處理中，請稍候..."):
            if not progress_window.winfo_exists():
                return
            stage_text.set(stage_name)
            status_var.set(status_message)

        def _init():
            try:
                # 第一階段：載入影片資料
                update_stage("載入影片資料", "正在讀取影片畫格資訊...")

                # 第二階段：初始化模型狀態
                self.after(1000, lambda: update_stage("初始化模型", "正在設定模型參數..."))

                # 第三階段：模型參數配置
                self.after(2000, lambda: update_stage("配置追蹤器", "正在設定追蹤參數..."))

                # 執行實際初始化
                self.inference_state = self.predictor.init_state(
                    video_path=self.video_dir.get())

                # 第四階段：完成
                self.after(500, lambda: update_stage("完成初始化", "初始化程序已完成!"))

                # 重置提示和遮罩
                self.prompts = {}
                self.current_masks = {}
                self.points = []
                self.labels = []
                self.current_object_id = 1
                self.obj_id_spinbox.set(1)

                # 關閉進度窗口並更新UI
                self.after(1000, progress_window.destroy)
                self.after(1000, self.on_predictor_initialized)

            except Exception as e:
                error_message = str(e)
                self.after(0, lambda: update_stage(
                    "初始化失敗", f"發生錯誤: {error_message[:50]}..."))
                self.after(2000, progress_window.destroy)
                self.after(0, lambda: messagebox.showerror(
                    "初始化錯誤", f"初始化 Predictor 狀態時出錯: {e}"))
                self.after(0, self.reset_ui_after_error)

        # 在背景線程中初始化
        threading.Thread(target=_init, daemon=True).start()

    def on_predictor_initialized(self):
        """Predictor 初始化成功後的回調"""
        self.update_status("追蹤狀態已初始化。可以開始標註。")
        self.set_ui_state(tk.NORMAL)
        self.enable_annotation_buttons(True)
        self.propagate_button.config(state=tk.NORMAL)
        self.update_frame_display()  # 重新繪製以清除舊遮罩

    # --- 畫格導覽與顯示 ---
    def update_frame_display(self, *args):
        """更新顯示的畫格影像和遮罩"""
        if not self.frame_names:
            return

        idx = self.current_frame_index.get()
        if 0 <= idx < len(self.frame_names):
            frame_path = os.path.join(
                self.video_dir.get(), self.frame_names[idx])
            try:
                self.current_image = Image.open(frame_path).convert("RGB")

                # --- 使用 Matplotlib 顯示 ---
                self.ax.clear()  # 清除之前的繪圖
                self.ax.imshow(self.current_image)
                # 使用英文標題
                self.ax.set_title(
                    f"Frame {idx} / {len(self.frame_names) - 1} (Object ID: {self.current_object_id})")
                self.ax.axis('off')  # 隱藏坐標軸

                # 繪製當前物件 ID 的提示點
                if self.current_object_id in self.prompts:
                    pts, lbls = self.prompts[self.current_object_id]
                    show_points_on_ax(pts, lbls, self.ax)

                # 繪製當前畫格的所有活動遮罩
                for obj_id, mask in self.current_masks.items():
                    show_mask_on_ax(mask, self.ax, obj_id=obj_id)

                # 繪製正在添加的點
                if self.points:
                    temp_points_np = np.array(self.points)
                    temp_labels_np = np.array(self.labels)
                    show_points_on_ax(temp_points_np, temp_labels_np, self.ax)

                self.canvas.draw_idle()  # 更新畫布

                # 更新畫格輸入框
                self.frame_entry.delete(0, tk.END)
                self.frame_entry.insert(0, str(idx))

            except Exception as e:
                messagebox.showerror("影像載入錯誤", f"無法載入或顯示畫格 {idx}: {e}")
                self.current_image = None
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """更新上一畫格/下一畫格按鈕的狀態"""
        idx = self.current_frame_index.get()
        total_frames = len(self.frame_names)
        self.prev_button.config(state=tk.NORMAL if idx > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if idx <
                                total_frames - 1 else tk.DISABLED)

    def prev_frame(self):
        """跳到上一畫格"""
        if self.is_processing:
            return
        idx = self.current_frame_index.get()
        if idx > 0:
            self.current_frame_index.set(idx - 1)
            # 清除當前畫格的臨時點和遮罩，因為換畫格了
            self.points = []
            self.labels = []
            self.current_masks = {}  # 換畫格後清除當前顯示的遮罩

    def next_frame(self):
        """跳到下一畫格"""
        if self.is_processing:
            return
        idx = self.current_frame_index.get()
        if idx < len(self.frame_names) - 1:
            self.current_frame_index.set(idx + 1)
            # 清除當前畫格的臨時點和遮罩
            self.points = []
            self.labels = []
            self.current_masks = {}  # 換畫格後清除當前顯示的遮罩

    # --- 標註互動 ---
    def on_canvas_click(self, event):
        """處理畫布點擊事件以添加提示點"""
        if self.is_processing:
            return
        if event.inaxes != self.ax:  # 確保點擊在影像區域內
            return
        if not self.predictor or not self.inference_state:
            messagebox.showwarning("警告", "請先載入模型並選擇影片目錄。")
            return

        x, y = int(event.xdata), int(event.ydata)

        # 左鍵: 正向點 (label=1), 右鍵: 反向點 (label=0)
        label = 1 if event.button == 1 else (0 if event.button == 3 else -1)

        if label != -1:
            self.points.append([x, y])
            self.labels.append(label)
            print(
                f"添加點: ({x}, {y}), 標籤: {label}, 物件 ID: {self.current_object_id}")

            # 如果啟用了自動套用，立即套用提示點
            if self.auto_apply.get():
                self.apply_prompts()
            else:
                # 僅預覽點
                self.update_frame_display()

    def update_object_id(self):
        """當 Spinbox 值改變時，更新當前物件 ID"""
        try:
            new_id = int(self.obj_id_spinbox.get())
            if new_id != self.current_object_id:
                self.current_object_id = new_id
                # 清除臨時點，因為換了物件
                self.points = []
                self.labels = []
                print(f"切換到物件 ID: {self.current_object_id}")
                self.update_frame_display()  # 更新標題和顯示的點
        except ValueError:
            # 如果輸入無效，恢復到目前 ID
            self.obj_id_spinbox.set(self.current_object_id)

    def apply_prompts(self):
        """將當前添加的點應用到模型"""
        if self.is_processing:
            messagebox.showwarning("警告", "正在處理中，請稍候。")
            return
        if not self.predictor or not self.inference_state:
            messagebox.showerror("錯誤", "模型或追蹤狀態未初始化。")
            return
        if not self.points:
            # 自動套用模式下不需要顯示此提示
            if not self.auto_apply.get():
                messagebox.showinfo("提示", "請先在影像上添加提示點。")
            return

        frame_idx = self.current_frame_index.get()
        obj_id = self.current_object_id
        points_np = np.array(self.points, dtype=np.float32)
        labels_np = np.array(self.labels, dtype=np.int32)

        # 將新點添加到 prompts 字典中（如果已有則更新）
        if obj_id in self.prompts:
            # 合併舊點和新點 (簡單合併，更複雜的邏輯可能需要)
            old_points, old_labels = self.prompts[obj_id]
            points_np = np.concatenate((old_points, points_np), axis=0)
            labels_np = np.concatenate((old_labels, labels_np), axis=0)

        self.prompts[obj_id] = (points_np, labels_np)

        self.update_status(f"正在為物件 {obj_id} 在畫格 {frame_idx} 添加提示點...")
        self.set_ui_state(tk.DISABLED)
        self.is_processing = True

        def _apply():
            try:
                # 使用 bfloat16 (如果 CUDA 可用)
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            points=points_np,
                            labels=labels_np,
                        )
                else:
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points_np,
                        labels=labels_np,
                    )

                # 更新當前畫格的遮罩顯示
                new_masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                self.after(0, lambda: self.on_prompts_applied(new_masks))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "應用提示點錯誤", f"處理提示點時發生錯誤: {e}"))
                self.after(0, self.reset_ui_after_error)
            finally:
                self.after(0, lambda: setattr(self, 'is_processing', False))

        threading.Thread(target=_apply, daemon=True).start()

    def on_prompts_applied(self, new_masks):
        """成功應用提示點後的回調"""
        self.update_status("提示點已應用，遮罩已更新。")
        self.set_ui_state(tk.NORMAL)
        self.points = []  # 清除臨時點
        self.labels = []
        self.current_masks = new_masks  # 更新要顯示的遮罩
        self.update_frame_display()  # 重新繪製以顯示新遮罩和合併後的點

    def clear_current_points(self):
        """清除當前畫格上尚未套用的點"""
        if self.is_processing:
            return
        self.points = []
        self.labels = []
        self.update_frame_display()
        self.update_status("已清除未套用的點。")

    def clear_object_annotation(self):
        """清除當前選定物件的所有標註（點和遮罩）"""
        if self.is_processing:
            return
        obj_id = self.current_object_id
        if obj_id in self.prompts:
            del self.prompts[obj_id]
            print(f"已清除物件 {obj_id} 的所有提示點。")
        if obj_id in self.current_masks:
            del self.current_masks[obj_id]
            print(f"已清除物件 {obj_id} 的當前遮罩。")

        # 注意：這裡沒有直接方法從 predictor 的 inference_state 中移除物件。
        messagebox.showinfo(
            "提示", f"已清除物件 {obj_id} 的本地標註記錄。\n若要從追蹤中完全移除，建議重新初始化追蹤狀態。")

        self.update_frame_display()
        self.update_status(f"已清除物件 {obj_id} 的本地標註。")

    # --- 影片處理 ---
    def run_propagation(self):
        """執行遮罩在影片中的推理並顯示詳細進度條"""
        if self.is_processing:
            messagebox.showwarning("警告", "正在處理中，請稍候。")
            return
        if not self.predictor or not self.inference_state:
            messagebox.showerror("錯誤", "模型或追蹤狀態未初始化。")
            return
        if not self.prompts:
            messagebox.showinfo("提示", "請至少為一個物件添加提示點。")
            return

        output_path = self.output_dir.get()
        if not output_path:
            messagebox.showerror("錯誤", "請先設定有效的輸出目錄。")
            return
        os.makedirs(output_path, exist_ok=True)  # 確保輸出目錄存在

        # 建立進度對話方塊
        progress_window = tk.Toplevel(self)
        progress_window.title("影片推理進度")
        progress_window.geometry("450x250")
        progress_window.resizable(False, False)
        progress_window.transient(self)  # 設為主視窗的子視窗
        progress_window.grab_set()  # 阻止與其他視窗互動

        # 置中顯示
        progress_window.geometry("+%d+%d" % (
            self.winfo_rootx() + (self.winfo_width() - 450) / 2,
            self.winfo_rooty() + (self.winfo_height() - 250) / 2))

        # 標題標籤
        main_label = ttk.Label(
            progress_window, text="正在執行影片遮罩推理", font=("", 12, "bold"))
        main_label.pack(pady=(20, 15))

        # 進度區域
        progress_frame = ttk.Frame(progress_window)
        progress_frame.pack(fill=tk.X, padx=20, pady=5)

        # 方向標籤
        direction_var = tk.StringVar(value="準備開始...")
        ttk.Label(progress_frame, textvariable=direction_var,
                  font=("", 10)).pack(anchor=tk.W, pady=(0, 5))

        # 總體進度條
        total_progress_label = ttk.Label(progress_frame, text="總體進度:")
        total_progress_label.pack(anchor=tk.W)

        total_progress = ttk.Progressbar(
            progress_frame, orient="horizontal", length=410, mode="determinate")
        total_progress.pack(fill=tk.X, pady=(0, 10))

        # 當前批次進度
        current_var = tk.StringVar(value="當前批次:")
        current_label = ttk.Label(progress_frame, textvariable=current_var)
        current_label.pack(anchor=tk.W)

        current_progress = ttk.Progressbar(
            progress_frame, orient="horizontal", length=410, mode="determinate")
        current_progress.pack(fill=tk.X, pady=(0, 10))

        # 詳細資訊
        stats_frame = ttk.Frame(progress_window)
        stats_frame.pack(fill=tk.X, padx=20)

        # 處理數據
        stats_var = tk.StringVar(value="已處理 0 / 0 畫格")
        stats_label = ttk.Label(stats_frame, textvariable=stats_var)
        stats_label.pack(side=tk.LEFT, pady=5)

        # 時間資訊
        time_var = tk.StringVar(value="估計剩餘時間: --:--")
        time_label = ttk.Label(stats_frame, textvariable=time_var)
        time_label.pack(side=tk.RIGHT, pady=5)

        # 詳細狀態資訊
        status_frame = ttk.Frame(progress_window)
        status_frame.pack(fill=tk.X, padx=20, pady=(5, 0))

        status_var = tk.StringVar(value="正在準備影片資料...")
        status_label = ttk.Label(
            status_frame, textvariable=status_var, font=("", 9, "italic"))
        status_label.pack(anchor=tk.W, pady=5)

        # 取消按鈕
        cancel_button = ttk.Button(progress_window, text="取消",
                                   command=lambda: setattr(self, '_cancel_propagation', True))
        cancel_button.pack(pady=15)

        # 初始化進度變數
        self.is_processing = True
        self._cancel_propagation = False
        total_frames = len(self.frame_names)
        processed_frames = 0
        start_time = time.time()

        # 開始處理
        self.update_status("正在執行影片推理...")

        def _propagate():
            nonlocal processed_frames
            try:
                # 使用 bfloat16 (如果 CUDA 可用)
                autocast_context = torch.autocast(
                    "cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()

                # 更新界面函數
                def update_progress(dir_name, frame_idx, dir_progress, total_progress_value):
                    if not progress_window.winfo_exists() or self._cancel_propagation:
                        return

                    # 計算時間估計
                    elapsed = time.time() - start_time
                    rate = processed_frames / elapsed if elapsed > 0 and processed_frames > 0 else 0
                    remaining = (total_frames * 2 -
                                 processed_frames) / rate if rate > 0 else 0

                    # 更新顯示
                    direction_var.set(
                        f"執行{dir_name}推理 ({int(dir_progress*100)}%)")
                    current_var.set(f"當前批次: 畫格 {frame_idx}")
                    stats_var.set(
                        f"已處理 {processed_frames} / {total_frames * 2} 畫格")

                    # 格式化剩餘時間
                    mins = int(remaining // 60)
                    secs = int(remaining % 60)
                    time_var.set(f"估計剩餘時間: {mins:02d}:{secs:02d}")

                    # 更新進度條
                    current_progress["value"] = dir_progress * 100
                    total_progress["value"] = total_progress_value

                    # 更新狀態
                    if frame_idx % 10 == 0:  # 每10幀更新一次狀態消息
                        status_var.set(
                            f"正在處理畫格 {frame_idx}，處理速度: {rate:.2f} 畫格/秒")

                with autocast_context:
                    # 執行正向和反向推理
                    for direction_idx, (direction_name, reverse_flag) in enumerate([("正向", False), ("反向", True)]):
                        if self._cancel_propagation:
                            raise Exception("用戶取消操作")

                        # 初始化批次處理計數
                        batch_frames = 0
                        direction_frames = 0
                        direction_start = processed_frames

                        # 開始推理
                        frame_iterator = self.predictor.propagate_in_video(
                            self.inference_state, reverse=reverse_flag)

                        # 獲取方向的總畫格數（用於進度計算）
                        direction_total = total_frames

                        for out_frame_idx, out_obj_ids, out_mask_logits in frame_iterator:
                            if self._cancel_propagation:
                                raise Exception("用戶取消操作")

                            # 如果該畫格有物件
                            if out_obj_ids:
                                # 儲存 ID 遮罩
                                _, mask_path = save_id_mask(
                                    out_mask_logits=out_mask_logits,
                                    ann_frame_dir=output_path,
                                    ann_frame_idx=out_frame_idx,
                                    show_visualization=False
                                )

                            # 更新進度計數
                            processed_frames += 1
                            direction_frames += 1
                            batch_frames += 1

                            # 計算進度百分比
                            dir_progress = direction_frames / direction_total
                            total_progress_value = (
                                processed_frames / (total_frames * 2)) * 100

                            # 每5畫格更新一次或在特定事件點更新UI
                            if batch_frames % 5 == 0 or direction_frames == 1 or direction_frames == direction_total:
                                self.after(0, lambda dn=direction_name, fi=out_frame_idx,
                                           dp=dir_progress, tp=total_progress_value:
                                           update_progress(dn, fi, dp, tp))

                        # 方向完成後的額外更新
                        self.after(0, lambda: status_var.set(
                            f"{direction_name}推理完成! 處理了 {direction_frames} 個畫格"))

                        # 短暫暫停，讓用戶看到階段完成
                        if direction_idx == 0:  # 第一個方向後
                            time.sleep(0.5)

                # 完成所有處理
                if self._cancel_propagation:
                    self.after(0, lambda: status_var.set("已取消影片推理"))
                    self.after(1000, progress_window.destroy)
                    self.after(0, lambda: self.update_status("影片推理已取消"))
                else:
                    self.after(0, lambda: status_var.set("影片推理已完成! 正在儲存結果..."))
                    self.after(0, lambda: total_progress.config(value=100))
                    self.after(2000, progress_window.destroy)
                    self.after(0, self.on_propagation_complete)

            except Exception as e:
                error_msg = str(e)
                # 如果是用戶取消，則顯示適當訊息
                if "用戶取消操作" in error_msg:
                    self.after(0, lambda: self.update_status("影片推理已取消"))
                else:
                    self.after(0, lambda: messagebox.showerror(
                        "推理錯誤", f"影片推理過程中發生錯誤: {e}"))
                    self.after(0, lambda: self.update_status("影片推理發生錯誤"))

                self.after(0, progress_window.destroy)
                self.after(0, self.reset_ui_after_error)

            finally:
                self.after(0, lambda: setattr(self, 'is_processing', False))
                self.after(0, lambda: setattr(
                    self, '_cancel_propagation', False))

        # 啟動背景處理線程
        threading.Thread(target=_propagate, daemon=True).start()

    def on_propagation_complete(self):
        """影片推理完成後的回調"""
        self.update_status(
            f"影片推理完成。ID 遮罩已儲存至 {self.output_dir.get()}/id_masks")
        self.set_ui_state(tk.NORMAL)

        # 顯示完成對話窗
        complete_dialog = tk.Toplevel(self)
        complete_dialog.title("推理完成")
        complete_dialog.geometry("400x200")
        complete_dialog.resizable(False, False)
        complete_dialog.transient(self)

        # 置中顯示
        complete_dialog.geometry("+%d+%d" % (
            self.winfo_rootx() + (self.winfo_width() - 400) / 2,
            self.winfo_rooty() + (self.winfo_height() - 200) / 2))

        # 成功圖標
        success_frame = ttk.Frame(complete_dialog)
        success_frame.pack(fill=tk.X, pady=(20, 10))

        # 文字使用綠色對勾符號代替實際圖標
        success_icon = ttk.Label(
            success_frame, text="✓", font=("", 24), foreground="green")
        success_icon.pack()

        # 完成訊息
        ttk.Label(complete_dialog, text="影片推理已完成！",
                  font=("", 14, "bold")).pack()
        ttk.Label(complete_dialog, text=f"遮罩已儲存至指定目錄:").pack(pady=(5, 0))
        ttk.Label(complete_dialog,
                  text=f"{self.output_dir.get()}/id_masks", font=("", 9, "italic")).pack()

        # 提示訊息
        ttk.Label(complete_dialog, text="建議重新初始化追蹤狀態以開始新的標註會話。",
                  wraplength=350).pack(pady=(15, 0))

        # 確認按鈕
        ttk.Button(complete_dialog, text="確定",
                   command=complete_dialog.destroy).pack(pady=15)

    # --- UI 狀態管理 ---
    def set_ui_state(self, state):
        """啟用或禁用介面元素"""
        for widget in self.winfo_children():
            self._set_widget_state(widget, state)
        # 特殊處理載入模型按鈕，只有在未載入時才啟用
        if self.predictor is None:
            self.load_model_button.config(
                state=tk.NORMAL if state == tk.NORMAL else tk.DISABLED)
        else:
            self.load_model_button.config(state=tk.DISABLED)

        # 確保導覽按鈕狀態正確
        if state == tk.NORMAL and self.frame_names:
            self.update_navigation_buttons()
        elif state == tk.DISABLED:
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)

        # 確保標註按鈕狀態正確
        if state == tk.NORMAL and self.predictor and self.inference_state:
            self.enable_annotation_buttons(True)
            self.propagate_button.config(state=tk.NORMAL)
        elif state == tk.DISABLED:
            self.enable_annotation_buttons(False)
            self.propagate_button.config(state=tk.DISABLED)

    def _set_widget_state(self, widget, state):
        """遞迴設置 Widget 狀態"""
        try:
            # 忽略 matplotlib canvas widget
            if isinstance(widget, (FigureCanvasTkAgg, tk.Canvas)):
                return
            widget.config(state=state)
        except tk.TclError:
            pass  # 忽略無法設定狀態的 Widget
        for child in widget.winfo_children():
            self._set_widget_state(child, state)

    def enable_annotation_buttons(self, enabled):
        """啟用或禁用標註相關按鈕"""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.add_points_button.config(state=state)
        self.clear_points_button.config(state=state)
        self.clear_object_button.config(state=state)
        self.obj_id_spinbox.config(state=state)

        # 自動套用選項
        self.auto_apply_check.config(state=state)

    def enable_navigation_buttons(self, enabled):
        """啟用或禁用畫格導覽按鈕"""
        if enabled and self.frame_names:
            self.update_navigation_buttons()
            self.goto_button.config(state=tk.NORMAL)
            self.frame_slider.config(state=tk.NORMAL)
            self.frame_entry.config(state=tk.NORMAL)
        else:
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            self.goto_button.config(state=tk.DISABLED)
            self.frame_slider.config(state=tk.DISABLED)
            self.frame_entry.config(state=tk.DISABLED)

    def update_status(self, message):
        """更新狀態欄訊息"""
        print(message)  # 同時打印到控制台
        self.status_bar.config(text=f"狀態: {message}")
        self.update_idletasks()  # 強制更新 UI

    def hide_progress_bar(self):
        """隱藏進度條"""
        if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
            self.progress_bar.pack_forget()


# --- 啟動應用程式 ---
if __name__ == "__main__":
    # --- 檢查相依性 ---
    try:
        import tkinter
        import PIL
        import matplotlib
        # sam2 已在頂部檢查
    except ImportError as e:
        messagebox.showerror(
            "缺少相依性", f"缺少必要的 Python 函式庫: {e.name}。\n請安裝所需的函式庫 (Tkinter, Pillow, Matplotlib)。")
        exit()

    # --- 執行 GUI ---
    app = SAM2_GUI()
    app.mainloop()


# --- 啟動應用程式 ---
if __name__ == "__main__":
    # --- 檢查相依性 ---
    try:
        import tkinter
        import PIL
        import matplotlib
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            messagebox.showerror("錯誤", "找不到 'sam2' 套件。請確保已正確安裝 SAM 2。")
            sys.exit(0)
    except ImportError as e:
        messagebox.showerror(
            "缺少相依性", f"缺少必要的 Python 函式庫: {e.name}。\n請安裝所需的函式庫 (Tkinter, Pillow, Matplotlib)。")
        sys.exit(0)

    # --- 顯示啟動畫面 ---
    splash = show_splash_screen()
    
    # --- 執行 GUI ---
    app = SAM2_GUI()
    
    # 確保啟動畫面關閉
    if splash and splash.winfo_exists():
        splash.destroy()
    
    app.mainloop()
    
    # 確保程序完全退出
    sys.exit(0)
