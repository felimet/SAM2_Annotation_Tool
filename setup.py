import PyInstaller.__main__
import os

# 獲取當前目錄的絕對路徑
current_dir = os.path.abspath(os.path.dirname(__file__))
favicon_path = os.path.join(current_dir, "favicon_io", "favicon.ico")

# 創建一個簡單的啟動畫面腳本
splash_script = """
# 啟動畫面自動關閉設置
import tkinter as tk
from PIL import Image, ImageTk
import sys
import os
import time

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 創建啟動畫面窗口
root = tk.Tk()
root.overrideredirect(True)  # 無邊框窗口

# 獲取屏幕尺寸以便置中顯示
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

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
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # 設置背景色
    root.configure(background='#1E3A8A')
    
    # 添加標誌圖像
    img_label = tk.Label(root, image=splash_img, bg='#1E3A8A')
    img_label.place(relx=0.5, rely=0.45, anchor='center')
    
    # 添加標題
    title_label = tk.Label(root, text="SAM2 標註工具", font=("Arial", 18, "bold"), 
                          fg='white', bg='#1E3A8A')
    title_label.place(relx=0.5, rely=0.75, anchor='center')
    
    # 添加載入中文字
    loading_label = tk.Label(root, text="正在載入，請稍候...", font=("Arial", 10), 
                            fg='white', bg='#1E3A8A')
    loading_label.place(relx=0.5, rely=0.85, anchor='center')
    
    # 更新界面
    root.update()
    
    # 設置自動關閉計時器 (3秒後自動關閉，或主程序啟動時關閉)
    root.after(3000, root.destroy)
    
except Exception as e:
    print(f"啟動畫面加載失敗: {e}")
    root.destroy()

# 執行啟動畫面
root.mainloop()
"""

# 將啟動畫面腳本寫入文件
with open("splash_screen.py", "w", encoding="utf-8") as f:
    f.write(splash_script)

# PyInstaller 命令行參數
PyInstaller.__main__.run([
    "GUI.py",  # 主腳本
    "--name=SAM2_Annotation_Tool",  # 生成的可執行文件名
    "--windowed",  # 不顯示命令行窗口
    f"--icon={favicon_path}",  # 應用圖示
    "--onedir",  # 生成目錄模式
    "--add-data=favicon_io;favicon_io",  # 添加圖示資源文件
    "--add-data=sam2;sam2",  # 添加SAM2模型相關文件
    "--add-binary=checkpoints;checkpoints",  # 添加模型權重
    "--clean",  # 清理臨時文件
    "--splash=splash_screen.py",  # 使用自定義啟動畫面腳本
    "--noconfirm",  # 不要詢問確認
    "--hidden-import=PIL._tkinter_finder",  # 隱式導入
    "--hidden-import=tkinter",
    "--hidden-import=matplotlib",
    "--hidden-import=torch",
    "--hidden-import=sam2"
])

# 清理臨時啟動畫面腳本
if os.path.exists("splash_screen.py"):
    os.remove("splash_screen.py")

print("封裝完成！可執行檔在 dist/SAM2_Annotation_Tool 目錄下。")