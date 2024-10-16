import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from statsmodels.tsa.ar_model import AutoReg
from itertools import permutations

@jit(nopython=True)
def is_match(x1, x2, length, time_series, r, delay):
    return np.max(np.abs(time_series[x1:x1+length*delay:delay] - time_series[x2:x2+length*delay:delay])) < r

@jit(nopython=True)
def count_overlaps(pairs, length, delay):
    overlap_count = 0
    for (i, j) in pairs:
        for (k, l) in pairs:
            if (i, j) == (k, l):
                continue
            if min(abs(i - k), abs(i - l), abs(j - k), abs(j - l)) <= length * delay:
                overlap_count += 1
    return overlap_count

@jit(nopython=True)
def calculate_sampen(time_series, m, r, delay):
    N = len(time_series)
    B = 0
    A = 0
    B_pairs = []
    A_pairs = []

    for i in range(N - m * delay):
        for j in range(i + 1, N - m * delay):
            if i != j and is_match(i, j, m, time_series, r, delay):
                B += 1
                B_pairs.append((i, j))
                if is_match(i, j, m + 1, time_series, r, delay):
                    A += 1
                    A_pairs.append((i, j))

    if B == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    CP = A / B
    SampEn = -np.log(CP)

    K_A = count_overlaps(np.array(A_pairs), m + 1, delay)
    K_B = count_overlaps(np.array(B_pairs), m, delay)

    SE_CP = np.sqrt((CP * (1 - CP) / B) + (1 / B**2) * (K_A + K_B * CP**2))
    SE_SampEn = SE_CP / CP

    return SampEn, SE_SampEn, CP, A, B

def find_optimal_r(time_series, m, r_values, delay):
    se_sampen_values = np.empty(len(r_values))
    for idx in range(len(r_values)):
        r = r_values[idx]
        try:
            _, se_sampen, _, _, _ = calculate_sampen(time_series, m, r, delay)
            se_sampen_values[idx] = se_sampen if not np.isnan(se_sampen) else np.inf
        except Exception as e:
            print(f"Error calculating SampEn for r={r}: {e}")
            se_sampen_values[idx] = np.inf

    optimal_r_index = np.argmin(se_sampen_values)
    optimal_r = r_values[optimal_r_index]

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

        self.file_path = ""
        self.ref_file_path = ""
        self.m_value = tk.IntVar(value=3)
        self.r_value = tk.DoubleVar(value=0.2)
        self.delay_value = tk.IntVar(value=1)
        self.num_windows = tk.IntVar(value=1)
        self.auto_m_value = tk.BooleanVar()
        self.auto_r_value = tk.BooleanVar()
        self.use_common_r_value = tk.BooleanVar()
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

        ttk.Label(frame, text="Time Delay:").grid(row=6, column=0, sticky=tk.W)
        self.delay_entry = ttk.Entry(frame, textvariable=self.delay_value)
        self.delay_entry.grid(row=6, column=1, sticky=(tk.W, tk.E))

        ttk.Label(frame, text="Number of Windows:").grid(row=7, column=0, sticky=tk.W)
        self.windows_entry = ttk.Entry(frame, textvariable=self.num_windows)
        self.windows_entry.grid(row=7, column=1, sticky=(tk.W, tk.E))

        ttk.Button(frame, text="Load CSV", command=self.load_csv).grid(row=8, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Load Reference CSV", command=self.load_ref_csv).grid(row=9, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Calculate Sample Entropy", command=self.calculate_sampen).grid(row=10, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Calculate Permutation Entropy", command=self.calculate_permen).grid(row=11, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Plot PermEn vs m", command=self.plot_permen_vs_m).grid(row=12, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Plot r values vs Window", command=self.plot_r_values_vs_window).grid(row=13, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(frame, text="")
        self.result_label.grid(row=14, column=0, columnspan=2, pady=10)

        for child in frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path = file_path
            messagebox.showinfo("File Loaded", f"Loaded file: {file_path}")

    def load_ref_csv(self):
        ref_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if ref_file_path:
            self.ref_file_path = ref_file_path
            messagebox.showinfo("Reference File Loaded", f"Loaded reference file: {ref_file_path}")

    def calculate_sampen(self):
        if not self.file_path or not self.ref_file_path:
            messagebox.showerror("Error", "Please load both CSV files first.")
            return

        try:
            time_series = pd.read_csv(self.file_path).values.flatten()
            ref_time_series = pd.read_csv(self.ref_file_path).values.flatten()
            print(f"Loaded time series of length {len(time_series)}")
            print(f"Loaded reference time series of length {len(ref_time_series)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            return

        if self.auto_m_value.get():
            m = fit_ar_model(time_series, max_lag=10)
            self.m_value.set(m)
        else:
            m = self.m_value.get()

        delay = self.delay_value.get()

        if len(time_series) < m * delay + 1 or len(ref_time_series) < m * delay + 1:
            messagebox.showerror("Error", "Time series is too short for the given template length and delay.")
            return

        num_windows = self.num_windows.get()

        if num_windows > 1:
            windows = divide_into_windows(time_series, num_windows)
            ref_windows = divide_into_windows(ref_time_series, num_windows)
        else:
            windows = [time_series]
            ref_windows = [ref_time_series]

        results = []
        ref_results = []
        self.r_values_per_window = []

        # Calculate standard deviations for both datasets
        std_dev1 = np.std(time_series)
        std_dev2 = np.std(ref_time_series)
        common_std_dev = (std_dev1 + std_dev2) / 2

        if self.auto_r_value.get():
            r_values = np.linspace(0.1 * common_std_dev, 0.25 * common_std_dev, 50)
            optimal_r, se_sampen_values = find_optimal_r(time_series, m, r_values, delay)
            self.r_value.set(optimal_r)
        else:
            optimal_r = self.r_value.get()

        for i, window in enumerate(windows):
            if self.use_common_r_value.get():
                r = optimal_r
            else:
                std_dev_window = np.std(window)
                r = 0.2 * std_dev_window

            self.r_values_per_window.append(r)

            try:
                sampen, se_sampen, cp, A, B = calculate_sampen(window, m, r, delay)
                results.append(sampen)
                ref_sampen, ref_se_sampen, ref_cp, ref_A, ref_B = calculate_sampen(ref_windows[i], m, r, delay)
                ref_results.append(ref_sampen)
            except IndexError as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
                return

        self.result_label.config(
            text=f"Sample Entropy Results:\n"
                 f"Windows: {results}\n"
                 f"Reference Windows: {ref_results}"
        )

        self.plot_sample_entropy(results, ref_results)

    def calculate_permen(self):
        if not self.file_path or not self.ref_file_path:
            messagebox.showerror("Error", "Please load both CSV files first.")
            return

        try:
            time_series = pd.read_csv(self.file_path).values.flatten()
            ref_time_series = pd.read_csv(self.ref_file_path).values.flatten()
            print(f"Loaded time series of length {len(time_series)}")
            print(f"Loaded reference time series of length {len(ref_time_series)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            return

        m = self.m_value.get()
        delay = self.delay_value.get()

        if len(time_series) < m * delay + 1 or len(ref_time_series) < m * delay + 1:
            messagebox.showerror("Error", "Time series is too short for the given template length and delay.")
            return

        num_windows = self.num_windows.get()

        if num_windows > 1:
            windows = divide_into_windows(time_series, num_windows)
            ref_windows = divide_into_windows(ref_time_series, num_windows)
        else:
            windows = [time_series]
            ref_windows = [ref_time_series]

        results = []
        ref_results = []

        for window in windows:
            try:
                permen = calculate_permutation_entropy(window, m, delay)
                results.append(permen)
            except Exception as e:
                results.append(np.nan)
                print(f"Error calculating PermEn: {e}")

        for ref_window in ref_windows:
            try:
                ref_permen = calculate_permutation_entropy(ref_window, m, delay)
                ref_results.append(ref_permen)
            except Exception as e:
                ref_results.append(np.nan)
                print(f"Error calculating Reference PermEn: {e}")

        self.result_label.config(
            text=f"Permutation Entropy Results:\n"
                 f"Windows: {results}\n"
                 f"Reference Windows: {ref_results}"
        )

        self.plot_permutation_entropy(results, ref_results)

    def plot_permen_vs_m(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        try:
            time_series = pd.read_csv(self.file_path).values.flatten()
            print(f"Loaded time series of length {len(time_series)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            return

        delay = self.delay_value.get()
        m_values = range(2, 10)  # Example range for m values
        permen_values = []

        for m in m_values:
            try:
                permen = calculate_permutation_entropy(time_series, m, delay)
                permen_values.append(permen)
            except Exception as e:
                permen_values.append(np.nan)
                print(f"Error calculating PermEn for m={m}: {e}")

        plt.figure(figsize=(10, 6))
        plt.plot(m_values, permen_values, marker='o')
        plt.xlabel('Template Length (m)')
        plt.ylabel('Permutation Entropy')
        plt.title(f'Permutation Entropy vs Template Length (m) with Delay={delay}')
        plt.show()

    def plot_sample_entropy(self, sampen_values, ref_sampen_values):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(sampen_values) + 1), sampen_values, marker='o', label='Sample Entropy')
        plt.plot(range(1, len(ref_sampen_values) + 1), ref_sampen_values, marker='x', label='Reference Sample Entropy')
        for i, r in enumerate(self.r_values_per_window):
            plt.annotate(f'r={r:.3f}', (i + 1, sampen_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Window Number')
        plt.ylabel('Sample Entropy')
        plt.title('Sample Entropy vs Window Number')
        plt.legend()
        plt.show()

    def plot_permutation_entropy(self, permen_values, ref_permen_values):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(permen_values) + 1), permen_values, marker='o', label='Permutation Entropy')
        plt.plot(range(1, len(ref_permen_values) + 1), ref_permen_values, marker='x', label='Reference Permutation Entropy')
        plt.xlabel('Window Number')
        plt.ylabel('Permutation Entropy')
        plt.title('Permutation Entropy vs Window Number')
        plt.legend()
        plt.show()

    def plot_se_vs_r(self, r_values, se_sampen_values, optimal_r):
        plt.figure(figsize=(10, 6))
        plt.plot(r_values, se_sampen_values, marker='o')
        plt.axvline(optimal_r, color='r', linestyle='--', label=f'Optimal r: {optimal_r:.3f}')
        plt.xlabel('r values')
        plt.ylabel('Standard Error of SampEn')
        plt.title(f'Standard Error of SampEn vs r values for m={self.m_value.get()}')
        plt.legend()
        plt.show()

    def plot_r_values_vs_window(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.r_values_per_window) + 1), self.r_values_per_window, marker='o')
        plt.xlabel('Window Number')
        plt.ylabel('r values')
        plt.title('r values vs Window Number')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = SampEnGUI(root)
    root.mainloop()
