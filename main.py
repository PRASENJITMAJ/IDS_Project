import tkinter as tk
from ASVM_preprocessing_gui import PreprocessingGUI
from ASVM_preprocessing_pipeline import Preprocessor

if __name__ == "__main__":
    root = tk.Tk()
    preprocessor = Preprocessor()
    app = PreprocessingGUI(root, preprocessor)
    root.mainloop()
