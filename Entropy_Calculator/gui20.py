import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from statsmodels.tsa.ar_model import AutoReg
from itertools import permutations
import EntropyHub as EH
from concurrent.futures import ThreadPoolExecutor

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

def calculate_permutation_entropy(time_series, m, delay):
    N = len(time_series)
    permutations_list = list(permutations(range(m)))
    perm_count = np.zeros(len(permutations_list))

    for i in range(N - m * delay + 1):
        sorted_index_tuple = tuple(np.argsort(time_series[i:i + m * delay:delay]))
        perm_index = permutations_list.index(sorted_index_tuple)
        perm_count[perm_index] += 1

    perm_prob = perm_count / np.sum(perm_count)
    perm_prob = perm_prob[perm_prob > 0]  # Remove zero probabilities
    PermEn = -np.sum(perm_prob * np.log(perm_prob))

    return PermEn

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

class SampEnGUI:
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

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Entropy Calculator", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(frame, text="Template Length (m):").grid(row=1, column=0, sticky=tk.W)
        self.m_entry = ttk.Entry(frame, textvariable=self.m_value)
        self.m_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))

        self.auto_m_check = ttk.Checkbutton(frame, text="Auto select m using AR model", variable=self.auto_m_value)
        self.auto_m_check.grid(row=2, column=0, columnspan=2)

        ttk.Label(frame, text="Tolerance (r):").grid(row=3, column=0, sticky=tk.W)
        self.r_entry = ttk.Entry(frame, textvariable=self.r_value)
        self.r_entry.grid(row=3, column=1, sticky=(tk.W, tk.E))

        self.auto_r_check = ttk.Checkbutton(frame, text="Auto select r", variable=self.auto_r_value)
        self.auto_r_check.grid(row=4, column=0, columnspan=2)

        self.use_common_r_check = ttk.Checkbutton(frame, text="Use common r value for all windows", variable=self.use_common_r_value)
        self.use_common_r_check.grid(row=5, column=0, columnspan=2)

        self.optimize_r_check = ttk.Checkbutton(frame, text="Optimize r value for each window", variable=self.optimize_r_per_window)
        self.optimize_r_check.grid(row=6, column=0, columnspan=2)

        ttk.Label(frame, text="Time Delay:").grid(row=7, column=0, sticky=tk.W)
        self.delay_entry = ttk.Entry(frame, textvariable=self.delay_value)
        self.delay_entry.grid(row=7, column=1, sticky=(tk.W, tk.E))

        ttk.Label(frame, text="Number of Windows:").grid(row=8, column=0, sticky=tk.W)
        self.windows_entry = ttk.Entry(frame, textvariable=self.num_windows)
        self.windows_entry.grid(row=8, column=1, sticky=(tk.W, tk.E))

        ttk.Button(frame, text="Load CSVs", command=self.load_csv).grid(row=9, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Add CSV", command=self.add_csv).grid(row=10, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Calculate Sample Entropy", command=self.calculate_sampen).grid(row=11, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Calculate Permutation Entropy", command=self.calculate_permen).grid(row=12, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Plot PermEn vs m", command=self.plot_permen_vs_m).grid(row=13, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Plot r values vs Window", command=self.plot_r_values_vs_window).grid(row=14, column=0, columnspan=2, pady=10)
        
        ttk.Button(frame, text="Download Results as CSV", command=self.download_results).grid(row=15, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(frame, text="")
        self.result_label.grid(row=16, column=0, columnspan=2, pady=10)

        for child in frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

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

    def calculate_sampen(self):
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

                    sampen_results = []
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

                        try:
                            sampen, se_sampen, cp, A, B, K_A, K_B = calculate_sampen(window, m, r, delay)
                            sampen_results.append((sampen, se_sampen, cp, A, B, K_A, K_B))
                            self.windows_data.append((file_path, i + 1, window))
                        except IndexError as e:
                            messagebox.showerror("Error", f"An error occurred in {file_path}: {e}")
                            return
                        except Exception as e:
                            messagebox.showerror("Error", f"Unexpected error in {file_path}: {e}")
                            return

                    self.r_values_per_window.append(r_values_per_window)
                    self.results.append((file_path, sampen_results))

            result_text = "Sample Entropy Results:\n"
            for file_path, sampen_results in self.results:
                result_text += f"{file_path}:\n{[res[0] for res in sampen_results]}\n"

            self.result_label.config(text=result_text)

            self.plot_sample_entropy(self.results)

        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            return

    def calculate_permen(self):
        if not self.file_paths:
            messagebox.showerror("Error", "Please load CSV files first.")
            return

        try:
            time_series_list = [pd.read_csv(file_path).values.flatten() for file_path in self.file_paths]
            for idx, time_series in enumerate(time_series_list):
                print(f"Loaded time series {idx + 1} of length {len(time_series)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV files: {e}")
            return

        results = []

        for file_path, time_series in zip(self.file_paths, time_series_list):
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

            permen_results = []

            for window in windows:
                try:
                    permen = calculate_permutation_entropy(window, m, delay)
                    permen_results.append(permen)
                except Exception as e:
                    permen_results.append(np.nan)
                    print(f"Error calculating PermEn in {file_path}: {e}")

            results.append((file_path, permen_results))

        result_text = "Permutation Entropy Results:\n"
        for file_path, permen_results in results:
            result_text += f"{file_path}:\n{permen_results}\n"

        self.result_label.config(text=result_text)

        self.plot_permutation_entropy(results)

    def plot_permen_vs_m(self):
        if not self.file_paths:
            messagebox.showerror("Error", "Please load CSV files first.")
            return

        try:
            time_series_list = [pd.read_csv(file_path).values.flatten() for file_path in self.file_paths]
            for idx, time_series in enumerate(time_series_list):
                print(f"Loaded time series {idx + 1} of length {len(time_series)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV files: {e}")
            return

        delay = self.delay_value.get()
        m_values = range(2, 10)  # Example range for m values

        for file_path, time_series in zip(self.file_paths, time_series_list):
            permen_values = []

            for m in m_values:
                try:
                    permen = calculate_permutation_entropy(time_series, m, delay)
                    permen_values.append(permen)
                except Exception as e:
                    permen_values.append(np.nan)
                    print(f"Error calculating PermEn for m={m} in {file_path}: {e}")

            plt.figure(figsize=(10, 6))
            plt.plot(m_values, permen_values, marker='o', label=f'{file_path}')
            plt.xlabel('Template Length (m)')
            plt.ylabel('Permutation Entropy')
            plt.title(f'Permutation Entropy vs Template Length (m) with Delay={delay}')
            plt.legend()
            plt.show()

    def plot_sample_entropy(self, results):
        plt.figure(figsize=(10, 6))
        for idx, (file_path, sampen_values) in enumerate(results):
            plt.plot(range(1, len(sampen_values) + 1), [val[0] for val in sampen_values], marker='o', label=file_path)
            if idx < len(self.r_values_per_window):
                for i, r in enumerate(self.r_values_per_window[idx]):
                    if i < len(sampen_values):
                        plt.annotate(f'r={r:.3f}', (i + 1, sampen_values[i][0]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Window Number')
        plt.ylabel('Sample Entropy')
        plt.title('Sample Entropy vs Window Number')
        plt.legend()
        plt.show()

    def plot_permutation_entropy(self, results):
        plt.figure(figsize=(10, 6))
        for file_path, permen_values in results:
            plt.plot(range(1, len(permen_values) + 1), permen_values, marker='o', label=file_path)
        plt.xlabel('Window Number')
        plt.ylabel('Permutation Entropy')
        plt.title('Permutation Entropy vs Window Number')
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

        for file_path, sampen_results in self.results:
            for i, (sampen, se_sampen, cp, A, B, K_A, K_B) in enumerate(sampen_results):
                for file, window_no, window_data in self.windows_data:
                    if file == file_path and window_no == i + 1:
                        data.append([
                            file_path, window_no, sampen, se_sampen, cp, A, B, K_A, K_B, window_data.tolist()
                        ])

        df = pd.DataFrame(data, columns=[
            "File Path", "Window", "Sample Entropy", "Standard Error", "CP Value", "A (Matches of m)", "B (Matches of m+1)", "K_A", "K_B", "Window Data"
        ])
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Results saved to {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SampEnGUI(root)
    root.mainloop()

