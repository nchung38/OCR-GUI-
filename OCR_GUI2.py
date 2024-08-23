import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import pytesseract
import threading
import time
import numpy as np
import pandas as pd
import pyperclip
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches


# OCR and Video Processing Code
def perform_ocr_on_video(video_path, roi, flip=False, rotate_angle=0):
    cap = cv2.VideoCapture(video_path)
    frames = [] 
    ret, frame = cap.read()
    while ret:
        if flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if rotate_angle:
            (h, w) = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
        frames.append(frame)
        ret, frame = cap.read()
    cap.release()

    mat = []
    for frame in frames:
        x, y, w, h = roi
        roi_image = frame[y:y+h, x:x+w]
        ocr_results = pytesseract.image_to_data(
            roi_image, output_type=pytesseract.Output.DATAFRAME,
            config='--psm 13 --oem 3 -l sev_seg4 -c tessedit_char_whitelist=0123456789.'
        )
        ocr_results = ocr_results[ocr_results.conf != -1]
        if not ocr_results.empty:
            text_results = ocr_results.iloc[0].text
            conf_results = ocr_results.iloc[0].conf

            # Ensure the text is a valid numerical value
            if text_results and isinstance(text_results, (int,float)) and float(conf_results) > 70:
                mat.append( float(text_results))
                print(f"Detected value: {text_results} with confidence: {conf_results}")
                frame_end = time.time()
                # print(f"time elapsed for frame {i}: {frame_end-frame_start}")
    return mat

# GUI Tool Class
class OCRToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Tool")

        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\NicholasChung\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


        # Sidebar (Left)
        self.sidebar = tk.Frame(self.root, width=200, bg="lightgray")
        self.sidebar.pack(expand=False, fill='y', side='left')

        # "Open" button to load video files
        self.open_button = tk.Button(self.sidebar, text="Open", command=self.load_videos)
        self.open_button.pack(padx=10, pady=10)

        # Dynamic list for selected videos
        self.video_list_frame = tk.Frame(self.sidebar)
        self.video_list_frame.pack(padx=10, pady=10)

        self.video_checkboxes = []
        self.video_flip_checkboxes = []
        self.video_rotate_checkboxes = []
        self.video_rotate_entries = []
        self.selected_videos = []
        self.roi_coords = []

        # Prepare videos button
        self.prepare_videos_button = tk.Button(self.sidebar, text="Prepare Videos", command = self.prepare_videos)
        self.prepare_videos_button.pack(side='bottom',padx =10, pady = 5)

        # "Read" button to perform OCR
        self.read_button = tk.Button(self.sidebar, text="Read", command=self.start_batch_processing)
        self.read_button.pack(side='bottom', padx=10, pady=5)
        

        # "Save" button to save results
        self.save_button = tk.Button(self.sidebar, text="Save", command=self.save_results)
        self.save_button.pack(side='bottom', padx=10, pady=5)

        # "View Data" button to display results
        self.view_data_button = tk.Button(self.sidebar, text="View Data", command=self.view_data)
        self.view_data_button.pack(side='bottom', padx=10, pady=5)


        # View Box (Right)
        self.view_box = tk.Frame(self.root, bg="white")
        self.view_box.pack(expand=True, fill='both', side='right')
        self.view_label = tk.Label(self.view_box, text="View Box", bg="white", font=("Arial", 12))
        self.view_label.pack(padx=10, pady=10)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")

        # Matplotlib canvas in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.view_box)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Data storage
        self.roi = None
        self.rect = None
        self.points = []  # Store the two points
        self.results = {}
        self.total_frames = 0
        self.start_time = 0
        self.processing = False

        # Connect the mouse events for ROI selection
        self.cid_click = self.canvas.mpl_connect('button_press_event', self.on_click)

    def load_videos(self):
        file_paths = filedialog.askopenfilenames(title="Select Videos", filetypes=[("Video files", "*.mp4 *.avi")])
        for file_path in file_paths:
            self.add_video_to_list(file_path)

    def add_video_to_list(self, video_path):
        video_name = os.path.basename(video_path)
        video_frame = tk.Frame(self.video_list_frame)
        video_frame.pack(fill='x', pady=5)

        # Video name label
        label = tk.Label(video_frame, text=video_name, width=20, anchor="w")
        label.pack(side='left')

        # Read checkbox
        read_var = tk.IntVar()
        read_checkbox = tk.Checkbutton(video_frame, text="Read", variable=read_var)
        read_checkbox.pack(side='left')
        read_checkbox.select()
        self.video_checkboxes.append(read_var)

        # Flip checkbox
        flip_var = tk.IntVar()
        flip_checkbox = tk.Checkbutton(video_frame, text="Flip", variable=flip_var)
        flip_checkbox.pack(side='left')
        flip_checkbox.select()
        self.video_flip_checkboxes.append(flip_var)

        # Rotate checkbox and entry
        rotate_var = tk.IntVar()
        rotate_checkbox = tk.Checkbutton(video_frame, text="Rotate", variable=rotate_var)
        rotate_checkbox.pack(side='left')
        self.video_rotate_checkboxes.append(rotate_var)

        rotate_entry = tk.Entry(video_frame, width=5)
        rotate_entry.pack(side='left')
        self.video_rotate_entries.append(rotate_entry)

        self.selected_videos.append(video_path)

    def start_batch_processing(self):
        if self.processing:
            return
        self.processing = True
        threading.Thread(target=self.process_videos).start()

    def prepare_videos(self):
        self.roi_coords = []  # Clear previous ROI coordinates
        for i, video in enumerate(self.selected_videos):
            if self.video_checkboxes[i].get():
                

                flip = self.video_flip_checkboxes[i].get() == 1
                rotate_angle = int(self.video_rotate_entries[i].get()) if self.video_rotate_checkboxes[i].get() else 0

                # Display first frame and get ROI
                cap = cv2.VideoCapture(video)
                ret, frame = cap.read()
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.total_frames += frame_count
                cap.release()
                if ret: 
                    frame = cv2.rotate(frame, cv2.ROTATE_180) if flip else frame
                    frame = self.rotate_frame(frame, rotate_angle)
                    self.get_roi(frame)

                    # Wait until the user has selected the ROI before proceeding to the next video
                    self.root.wait_variable(self.roi_var)  # Pause until ROI is selected for this video
        time.sleep(0.6)
        self.ax.clear()
        self.ax.axis("off")
        # self.rect.remove()
        self.canvas.draw()
        self.update_view_box("Videos are prepared. Press Read to begin processing.")


    def process_videos(self):
        start_time = time.time()
        estimated_time = (self.total_frames * 0.5) / 60
        self.update_view_box(f"Processing {len(self.selected_videos)} videos. Estimated time: {estimated_time:.1f} minutes.\n")
        for i, video in enumerate(self.selected_videos):
            if self.video_checkboxes[i].get():
                flip = self.video_flip_checkboxes[i].get() == 1
                rotate_angle = int(self.video_rotate_entries[i].get()) if self.video_rotate_checkboxes[i].get() else 0

                if self.roi:
                    # Process video with OCR
                    curr_roi = self.roi_coords[i]
                    mat = perform_ocr_on_video(video, curr_roi, flip, rotate_angle)
                    self.results[video] = mat   

        self.update_view_box("Batch finished. Press Save to save data, or View Data to view as a list.")
        self.processing = False

    def rotate_frame(self, frame, angle):
        (h, w) = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h))

    def get_roi(self, frame):
        # Display the frame using Matplotlib
        roi_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.imshow(roi_im)
        self.ax.axis('off')
        self.canvas.draw()

        # Wait for the user to select ROI using clicks
        self.update_view_box("Click on two points to select the ROI")
        self.roi_var = tk.IntVar()  # Variable to track ROI selection

    def on_click(self, event):
        if event.inaxes:  # Check if the click is inside the axes
            # Capture the point and add it to the points list
            self.points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'r+')  # Draw a red point ('ro') at the clicked position
            self.canvas.draw()
            if len(self.points) == 1:
                # On the first click, just record the point
                print(f"First point selected: {self.points[0]}")
            elif len(self.points) == 2:
                # On the second click, complete the ROI selection
                print(f"Second point selected: {self.points[1]}")
                self.create_rectangle()
                self.roi_var.set(1)  # Set the variable to indicate ROI selection is done
    
    def create_rectangle(self):
        if len(self.points) == 2:
            x0, y0 = self.points[0]
            x1, y1 = self.points[1]
            self.roi = [int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0))]
            self.roi_coords.append(self.roi)

            # If a rectangle already exists, remove it
            # if self.rect:
            #     self.rect.remove()

            # # Draw the rectangle
            # self.rect = patches.Rectangle((self.roi[0], self.roi[1]), self.roi[2], self.roi[3],
            #                             linewidth=2, edgecolor='r', facecolor='none')
            # self.ax.add_patch(self.rect)
            # self.canvas.draw()

            # Clear the points after drawing the rectangle
            self.points = []

    def update_view_box(self, text):
        self.view_label.config(text=text)

    def save_results(self):
        if not self.results:
            messagebox.showwarning("No Data", "No OCR results to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            # Convert the results dictionary to a DataFrame, ensuring all columns have the same length
            max_len = max(len(v) for v in self.results.values())
            padded_results = {k: v + [np.nan] * (max_len - len(v)) for k, v in self.results.items()}
            
            df = pd.DataFrame(padded_results)
            df.to_excel(file_path, index=False)  # Save without the index column
            messagebox.showinfo("Success", f"Results saved to {file_path}.")

    def view_data(self):
        if not self.results:
            messagebox.showwarning("No Data", "No OCR results to display.")
            return

        # Show data in view box
        result_str = ""
        for video, data in self.results.items():
            result_str += f"{os.path.basename(video)}: {', '.join(map(str, data))}\n"
        self.update_view_box(result_str)


# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = OCRToolGUI(root)
    root.geometry("800x600")
    root.mainloop()
