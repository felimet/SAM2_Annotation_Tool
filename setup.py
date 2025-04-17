import PyInstaller.__main__
import os
import sys

# 獲取當前目錄的絕對路徑
current_dir = os.path.abspath(os.path.dirname(__file__))
favicon_path = os.path.join(current_dir, "favicon_io", "favicon.ico")
splash_path = os.path.join(current_dir, "favicon_io", "android-chrome-512x512.png")

# 修改 GUI.py 添加啟動畫面處理代碼
gui_file = os.path.join(current_dir, "GUI.py")
with open(gui_file, "r", encoding="utf-8") as f:
    gui_content = f.read()

# 檢查是否已經添加了啟動畫面代碼
if "def show_splash_screen():" not in gui_content:
    # 在 import 部分之後添加啟動畫面代碼
    splash_code = '''
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
'''

    # 修改 __main__ 部分以添加啟動畫面調用
    main_code = '''
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
            "缺少相依性", f"缺少必要的 Python 函式庫: {e.name}。\\n請安裝所需的函式庫 (Tkinter, Pillow, Matplotlib)。")
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
'''

    # 找到原始的 __main__ 部分並替換
    import re
    main_pattern = r'# --- 啟動應用程式 ---\s*if __name__ == "__main__":.+?sys\.exit\(0\)'
    if re.search(main_pattern, gui_content, re.DOTALL):
        gui_content = re.sub(main_pattern, main_code, gui_content, flags=re.DOTALL)
    else:
        # 如果沒找到匹配模式，添加到文件末尾
        gui_content = gui_content.rstrip() + "\n\n" + main_code

    # 在適當位置插入啟動畫面代碼
    import_end_idx = gui_content.find("# --- 從原始腳本導入或重新定義必要的函數和設定 ---")
    if import_end_idx > 0:
        gui_content = gui_content[:import_end_idx] + splash_code + "\n\n" + gui_content[import_end_idx:]
    else:
        # 如果找不到特定位置，添加到 import 部分之後
        import_section_end = max(gui_content.find("\n\n", gui_content.find("import")), 0)
        gui_content = gui_content[:import_section_end] + "\n" + splash_code + gui_content[import_section_end:]

    # 寫回修改後的內容
    with open(gui_file, "w", encoding="utf-8") as f:
        f.write(gui_content)

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
    # "--splash=splash_screen.py",  # 移除錯誤的啟動畫面設置
    "--noconfirm",  # 不要詢問確認
    "--hidden-import=PIL._tkinter_finder",  # 隱式導入
    "--hidden-import=tkinter",
    "--hidden-import=matplotlib",
    "--hidden-import=torch",
    "--hidden-import=sam2"
])

print("封裝完成！可執行檔在 dist/SAM2_Annotation_Tool 目錄下。")