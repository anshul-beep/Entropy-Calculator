import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from statsmodels.tsa.ar_model import AutoReg
from concurrent.futures import ThreadPoolExecutor
import EntropyHub as EH

CHUNK_SIZE = 10000  # Define the chunk size for processing large files

def calculate_sampen(time_series, m, r, delay):
    try:
        result, A, B, (Vcp, a, b) = EH.SampEn(time_series, m, delay, r, Vcp=True)
        SampEn = result[-1]
        CP = A[-1] / B[-1]
        SE_SampEn = Vcp / CP
        K_A = a
        K_B = b
        return SampEn, SE_SampEn, CP, A[-1], B[-1], K_A, K_B
    except Exception as e:
        print(f"Error in calculate_sampen: {e}")
        raise

def calculate_permen(time_series, m, delay, perm_type, norm):
    try:
        PermEn, Pnorm, cPE = EH.PermEn(time_series, m=m, tau=delay, Typex=perm_type, Norm=norm)
        return PermEn[-1], Pnorm[-1], cPE[-1]  # Return only the values for m=m
    except Exception as e:
        print(f"Error in calculate_permen: {e}")
        raise

def find_optimal_r(time_series, m, r_values, delay):
    def calculate_r_value(r):
        try:
            _, se_sampen, _, _, _, _, _ = calculate_sampen(time_series, m, r, delay)
            return se_sampen if not np.isnan(se_sampen) else np.inf
        except Exception as e:
            print(f"Error calculating SampEn for r={r}: {e}")
            return np.inf

    with ThreadPoolExecutor() as executor:
        se_sampen_values = list(executor.map(calculate_r_value, r_values))

    optimal_r_index = np.argmin(se_sampen_values)
    optimal_r = r_values[optimal_r_index]

    print(f"Optimal r value: {optimal_r}")
    return optimal_r, se_sampen_values

def fit_ar_model(time_series, max_lag):
    aic_values = []
    for lag in range(1, max_lag + 1):
        model = AutoReg(time_series, lags=lag).fit()
        aic_values.append(model.aic)
    best_aic_lag = np.argmin(aic_values) + 1
    return best_aic_lag

def divide_into_windows(time_series, num_windows):
    window_size = len(time_series) // num_windows

    # Remove first 50% of data from the first window
    start_index = window_size // 2
    # Remove last 50% of data from the last window
    end_index = len(time_series) - (window_size // 2)

    remaining_data = time_series[start_index:end_index]

    new_window_size = len(remaining_data) // num_windows
    windows = [remaining_data[i * new_window_size:(i + 1) * new_window_size] for i in range(num_windows)]

    return windows

class EntropyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Entropy Calculator")

        self.file_paths = []
        self.m_value = tk.IntVar(value=3)
        self.r_value = tk.DoubleVar(value=0.2)
        self.delay_value = tk.IntVar(value=1)
        self.num_windows = tk.IntVar(value=1)
        self.auto_m_value = tk.BooleanVar()
        self.auto_r_value = tk.BooleanVar()
        self.use_common_r_value = tk.BooleanVar()
        self.optimize_r_per_window = tk.BooleanVar(value=False)
        self.r_values_per_window = []
        self.entropy_type = tk.StringVar(value="Sample Entropy")
        self.perm_entropy_type = tk.StringVar(value="none")
        self.norm_value = tk.BooleanVar()

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Entropy Calculator", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(frame, text="Select Entropy Type:").grid(row=1, column=0, sticky=tk.W)
        self.entropy_menu = ttk.Combobox(frame, textvariable=self.entropy_type, values=["Sample Entropy", "Permutation Entropy"])
        self.entropy_menu.grid(row=1, column=1, sticky=(tk.W, tk.E))
        self.entropy_menu.bind("<<ComboboxSelected>>", self.update_gui)

        ttk.Label(frame, text="Template Length (m):").grid(row=2, column=0, sticky=tk.W)
        self.m_entry = ttk.Entry(frame, textvariable=self.m_value)
        self.m_entry.grid(row=2, column=1, sticky=(tk.W, tk.E))

        self.auto_m_check = ttk.Checkbutton(frame, text="Auto select m using AR model", variable=self.auto_m_value)
        self.auto_m_check.grid(row=3, column=0, columnspan=2)

        self.r_label = ttk.Label(frame, text="Tolerance (r):")
        self.r_label.grid(row=4, column=0, sticky=tk.W)
        self.r_entry = ttk.Entry(frame, textvariable=self.r_value)
        self.r_entry.grid(row=4, column=1, sticky=(tk.W, tk.E))

        self.auto_r_check = ttk.Checkbutton(frame, text="Auto select r", variable=self.auto_r_value)
        self.auto_r_check.grid(row=5, column=0, columnspan=2)

        self.use_common_r_check = ttk.Checkbutton(frame, text="Use common r value for all windows", variable=self.use_common_r_value)
        self.use_common_r_check.grid(row=6, column=0, columnspan=2)

        self.optimize_r_check = ttk.Checkbutton(frame, text="Optimize r value for each window", variable=self.optimize_r_per_window)
        self.optimize_r_check.grid(row=7, column=0, columnspan=2)

        ttk.Label(frame, text="Time Delay:").grid(row=8, column=0, sticky=tk.W)
        self.delay_entry = ttk.Entry(frame, textvariable=self.delay_value)
        self.delay_entry.grid(row=8, column=1, sticky=(tk.W, tk.E))

        ttk.Label(frame, text="Number of Windows:").grid(row=9, column=0, sticky=tk.W)
        self.windows_entry = ttk.Entry(frame, textvariable=self.num_windows)
        self.windows_entry.grid(row=9, column=1, sticky=(tk.W, tk.E))

        self.perm_entropy_menu_label = ttk.Label(frame, text="Select Permutation Entropy Type:")
        self.perm_entropy_menu_label.grid(row=10, column=0, sticky=tk.W)
        self.perm_entropy_menu = ttk.Combobox(frame, textvariable=self.perm_entropy_type, values=["none", "uniquant", "finegrain", "modified", "ampaware", "weighted", "edge", "phase"])
        self.perm_entropy_menu.grid(row=10, column=1, sticky=(tk.W, tk.E))

        self.norm_check = ttk.Checkbutton(frame, text="Normalize", variable=self.norm_value)
        self.norm_check.grid(row=11, column=0, columnspan=2)

        ttk.Button(frame, text="Load CSVs", command=self.load_csv).grid(row=12, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Add CSV", command=self.add_csv).grid(row=13, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Calculate Entropy", command=self.calculate_entropy).grid(row=14, column=0, columnspan=2, pady=10)
        
        ttk.Button(frame, text="Plot r values vs Window", command=self.plot_r_values_vs_window).grid(row=15, column=0, columnspan=2, pady=10)
        
        ttk.Button(frame, text="Download Results as CSV", command=self.download_results).grid(row=16, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(frame, text="")
        self.result_label.grid(row=17, column=0, columnspan=2, pady=10)

        for child in frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.update_gui(None)

    def update_gui(self, event):
        if self.entropy_type.get() == "Permutation Entropy":
            self.r_label.grid_remove()
            self.r_entry.grid_remove()
            self.auto_r_check.grid_remove()
            self.use_common_r_check.grid_remove()
            self.optimize_r_check.grid_remove()
            self.perm_entropy_menu_label.grid()
            self.perm_entropy_menu.grid()
        else:
            self.r_label.grid()
            self.r_entry.grid()
            self.auto_r_check.grid()
            self.use_common_r_check.grid()
            self.optimize_r_check.grid()
            self.perm_entropy_menu_label.grid_remove()
            self.perm_entropy_menu.grid_remove()

    def load_csv(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            self.file_paths = list(file_paths)
            messagebox.showinfo("Files Loaded", f"Loaded files: {', '.join(file_paths)}")

    def add_csv(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            self.file_paths.extend(file_paths)
            messagebox.showinfo("Files Added", f"Added files: {', '.join(file_paths)}")

    def process_chunk(self, chunk, m, r, delay):
        return [calculate_sampen(chunk[start:start + CHUNK_SIZE], m, r, delay) for start in range(0, len(chunk), CHUNK_SIZE)]

    def calculate_entropy(self):
        if not self.file_paths:
            messagebox.showerror("Error", "Please load CSV files first.")
            return

        try:
            self.results = []
            self.windows_data = []
            self.r_values_per_window = []

            for file_path in self.file_paths:
                for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
                    time_series = chunk.values.flatten()
                    print(f"Loaded chunk of length {len(time_series)}")

                    if self.auto_m_value.get():
                        m = fit_ar_model(time_series, max_lag=10)
                        self.m_value.set(m)
                    else:
                        m = self.m_value.get()

                    delay = self.delay_value.get()

                    if len(time_series) < m * delay + 1:
                        messagebox.showerror("Error", f"Time series in {file_path} is too short for the given template length and delay.")
                        return

                    num_windows = self.num_windows.get()

                    if num_windows > 1:
                        windows = divide_into_windows(time_series, num_windows)
                    else:
                        windows = [time_series]

                    entropy_results = []
                    r_values_per_window = []

                    # Calculate standard deviation for the dataset
                    std_dev = np.std(time_series)

                    if self.auto_r_value.get():
                        r_values = np.linspace(0.1 * std_dev, 0.25 * std_dev, 10)  # Reduce the number of r values to test
                        optimal_r, se_sampen_values = find_optimal_r(time_series, m, r_values, delay)
                        self.r_value.set(optimal_r)
                    else:
                        optimal_r = self.r_value.get()

                    for i, window in enumerate(windows):
                        if self.entropy_type.get() == "Sample Entropy":
                            if self.use_common_r_value.get():
                                r = optimal_r
                            elif self.optimize_r_per_window.get():
                                std_dev_window = np.std(window)
                                r_values = np.linspace(0.1 * std_dev_window, 0.25 * std_dev_window, 10)  # Reduce the number of r values to test
                                r, _ = find_optimal_r(window, m, r_values, delay)
                            else:
                                std_dev_window = np.std(window)
                                r = 0.2 * std_dev_window

                            r_values_per_window.append(r)
                            entropy_result = calculate_sampen(window, m, r, delay)
                        elif self.entropy_type.get() == "Permutation Entropy":
                            entropy_result = calculate_permen(window, m, delay, self.perm_entropy_type.get(), self.norm_value.get())

                        entropy_results.append(entropy_result)
                        self.windows_data.append((file_path, i + 1, window))

                    if self.entropy_type.get() == "Sample Entropy":
                        self.r_values_per_window.append(r_values_per_window)
                    self.results.append((file_path, entropy_results))

            result_text = f"{self.entropy_type.get()} Results:\n"
            for file_path, entropy_results in self.results:
                result_text += f"{file_path}:\n"
                for result in entropy_results:
                    if isinstance(result, tuple):
                        result_text += f"Entropy: {result[0]}, Normalized Entropy: {result[1]}, Conditional Entropy: {result[2]}\n"
                    else:
                        result_text += f"{result}\n"

            self.result_label.config(text=result_text)

            self.plot_entropy(self.results)

        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            return

    def plot_entropy(self, results):
        plt.figure(figsize=(10, 6))
        for idx, (file_path, entropy_values) in enumerate(results):
            plt.plot(range(1, len(entropy_values) + 1), [val[0] if isinstance(val, tuple) else val for val in entropy_values], marker='o', label=file_path)
            if self.entropy_type.get() == "Sample Entropy" and idx < len(self.r_values_per_window):
                for i, r in enumerate(self.r_values_per_window[idx]):
                    if i < len(entropy_values):
                        plt.annotate(f'r={r:.3f}', (i + 1, entropy_values[i][0] if isinstance(entropy_values[i], tuple) else entropy_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Window Number')
        plt.ylabel(f'{self.entropy_type.get()}')
        plt.title(f'{self.entropy_type.get()} vs Window Number')
        plt.legend()
        plt.show()

    def plot_r_values_vs_window(self):
        plt.figure(figsize=(10, 6))
        for r_values in self.r_values_per_window:
            plt.plot(range(1, len(r_values) + 1), r_values, marker='o')
        plt.xlabel('Window Number')
        plt.ylabel('r values')
        plt.title('r values vs Window Number')
        plt.show()

    def download_results(self):
        if not hasattr(self, 'results') or not self.results:
            messagebox.showerror("Error", "No results to download. Please calculate entropies first.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return

        data = []

        for file_path, entropy_results in self.results:
            for i, result in enumerate(entropy_results):
                for file, window_no, window_data in self.windows_data:
                    if file == file_path and window_no == i + 1:
                        if isinstance(result, tuple):
                            data.append([
                                file_path, window_no, result[0], result[1], result[2], window_data.tolist()
                            ])
                        else:
                            data.append([
                                file_path, window_no, result, np.nan, np.nan, window_data.tolist()
                            ])

        df = pd.DataFrame(data, columns=[
            "File Path", "Window", "Entropy", "Normalized Entropy", "Conditional Entropy", "Window Data"
        ])
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Results saved to {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EntropyGUI(root)
    root.mainloop()





