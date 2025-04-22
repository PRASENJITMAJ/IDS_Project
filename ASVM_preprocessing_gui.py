import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class PreprocessingGUI:
    def __init__(self, root, preprocessing):
        self.root = root
        self.root.title("Network Traffic Data Preprocessor")
        self.root.geometry("1200x800")

        self.preprocessing = preprocessing
        self.df = None
        self.processed_df = None

        self.create_gui()

    def create_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(main_frame, text="Load CSV", command=self.load_csv).grid(row=0, column=0, pady=5)

        process_frame = ttk.LabelFrame(main_frame, text="Processing Steps", padding="5")
        process_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))

        ttk.Button(process_frame, text="1. Clean Data", command=self.clean_data).grid(row=0, column=0, padx=5)
        ttk.Button(process_frame, text="2. Encode Labels", command=self.encode_labels).grid(row=0, column=1, padx=5)
        ttk.Button(process_frame, text="3. Remove Features", command=self.remove_features).grid(row=0, column=2, padx=5)
        ttk.Button(process_frame, text="4. Normalize", command=self.normalize_features).grid(row=0, column=3, padx=5)
        ttk.Button(process_frame, text="5. Train Autoencoder", command=self.train_autoencoder).grid(row=0, column=4, padx=5)
        ttk.Button(process_frame, text="6. Save CSV", command=self.save_csv).grid(row=0, column=5, padx=5)

        self.status_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=2, column=0, pady=5)

        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="5")
        preview_frame.grid(row=3, column=0, pady=5, sticky=(tk.W, tk.E))

        self.preview_text = tk.Text(preview_frame, height=10, width=100)
        self.preview_text.grid(row=0, column=0, pady=5)

        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="5")
        viz_frame.grid(row=4, column=0, pady=5, sticky=(tk.W, tk.E))

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, pady=5)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = self.preprocessing.load_csv(file_path)
                self.status_var.set(f"Loaded CSV with shape: {self.df.shape}")
                self.update_preview()
                self.visualize_data()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading CSV: {str(e)}")

    def clean_data(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return
        try:
            self.processed_df = self.preprocessing.clean_data(self.df)
            self.status_var.set("Cleaned data")
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Error cleaning data: {str(e)}")

    def encode_labels(self):
        if self.processed_df is None:
            messagebox.showwarning("Warning", "Please clean the data first")
            return
        try:
            self.processed_df = self.preprocessing.encode_labels(self.processed_df)
            self.status_var.set("Encoded categorical features")
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Error encoding labels: {str(e)}")

    def remove_features(self):
        if self.processed_df is None:
            messagebox.showwarning("Warning", "Please encode labels first")
            return
        try:
            self.processed_df = self.preprocessing.remove_features(self.processed_df)
            self.status_var.set("Removed irrelevant features")
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Error removing features: {str(e)}")

    def normalize_features(self):
        if self.processed_df is None:
            messagebox.showwarning("Warning", "Please remove features first")
            return
        try:
            self.processed_df = self.preprocessing.normalize_features(self.processed_df)
            self.status_var.set("Normalized numeric features")
            self.update_preview()
            self.visualize_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error normalizing features: {str(e)}")

    def train_autoencoder(self):
        if self.processed_df is None:
            messagebox.showwarning("Warning", "Please normalize features first")
            return
        try:
            self.processed_df = self.preprocessing.train_autoencoder(self.processed_df)
            self.status_var.set("Trained autoencoder")
            self.update_preview()
            self.visualize_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error training autoencoder: {str(e)}")

    def save_csv(self):
        if self.processed_df is None:
            messagebox.showwarning("Warning", "No processed data available to save")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.preprocessing.save_csv(self.processed_df, file_path)
                messagebox.showinfo("Success", f"Processed data saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving file: {str(e)}")

    def update_preview(self):
        df = self.processed_df if self.processed_df is not None else self.df
        if df is not None:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, df.head().to_string())

    def visualize_data(self):
        if self.processed_df is None:
            return
        self.ax.clear()
        numeric_cols = self.processed_df.select_dtypes(include=['float64', 'int64']).columns[:5]
        self.processed_df[numeric_cols].boxplot(ax=self.ax)
        self.ax.set_title("Feature Distribution")
        self.ax.tick_params(axis='x', rotation=45)
        self.canvas.draw()