import tkinter as tk
from PCA_Preprocessing_GuI import PreprocessingGUI
from PCA_preprocessing import Preprocessor

if __name__ == "__main__":
    root = tk.Tk()
    preprocessor = Preprocessor()
    app = PreprocessingGUI(root, preprocessor)
    root.mainloop()