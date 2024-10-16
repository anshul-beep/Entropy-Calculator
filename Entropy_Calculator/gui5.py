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

    for i in range(N - m * delay):
        sorted_index_tuple = tuple(np.argsort(time_series[i:i + m * delay:delay]))
        perm_index = permutations_list.index(sorted_index_tuple)
        perm_count[perm_index] += 1

    perm_prob = perm_count / np.sum(perm_count)
    perm_prob = perm_prob[perm_prob > 0]  # Remove zero probabilities
    PermEn = -np.sum(perm_prob * np.log(perm_prob))

    return PermEn

class SampEnGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Entropy Calculator")

        self.file_path = ""
        self.ref_file_path = ""
        self.m_value = tk.IntVar(value=2)
        self.r_value = tk.DoubleVar(value=0.2)
        self.delay_value = tk.IntVar(value=1)
        self.auto_m_value = tk.BooleanVar()
        self.auto_r_value = tk.BooleanVar()

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

        ttk.Label(frame, text="Time Delay:").grid(row=5, column=0, sticky=tk.W)
        self.delay_entry = ttk.Entry(frame, textvariable=self.delay_value)
        self.delay_entry.grid(row=5, column=1, sticky=(tk.W, tk.E))

        ttk.Button(frame, text="Load CSV", command=self.load_csv).grid(row=6, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Load Reference CSV", command=self.load_ref_csv).grid(row=7, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Calculate Sample Entropy", command=self.calculate_sampen).grid(row=8, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Calculate Permutation Entropy", command=self.calculate_permen).grid(row=9, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(frame, text="")
        self.result_label.grid(row=10, column=0, columnspan=2, pady=10)

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
        if not self.file_path:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        try:
            time_series = pd.read_csv(self.file_path).values.flatten()
            print(f"Loaded time series of length {len(time_series)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            return

        if self.auto_m_value.get():
            m = fit_ar_model(time_series, max_lag=10)
            self.m_value.set(m)
        else:
            m = self.m_value.get()

        delay = self.delay_value.get()

        if len(time_series) < m * delay + 1:
            messagebox.showerror("Error", "Time series is too short for the given template length and delay.")
            return

        if self.auto_r_value.get():
            std_dev = np.std(time_series)
            r_min = 0.1 * std_dev
            r_max = 0.25 * std_dev
            r_values = np.linspace(r_min, r_max, 50)
            try:
                optimal_r, se_sampen_values = find_optimal_r(time_series, m, r_values, delay)
                self.r_value.set(optimal_r)
                sampen, se_sampen, cp, A, B = calculate_sampen(time_series, m, optimal_r, delay)
                self.result_label.config(text=f"Optimal m={m}, Optimal r={optimal_r:.3f}, Delay={delay}\nSampEn: {sampen:.3f}\nCP: {cp:.3f}\nA: {A}\nB: {B}")
                self.plot_se_vs_r(r_values, se_sampen_values, optimal_r)
            except IndexError as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
        else:
            r = self.r_value.get()
            try:
                sampen, se_sampen, cp, A, B = calculate_sampen(time_series, m, r, delay)
                if np.isnan(sampen):
                    messagebox.showerror("Error", "Failed to calculate Sample Entropy.")
                else:
                    self.result_label.config(text=f"SampEn for m={m}, r={r:.3f}, Delay={delay}\nSampEn: {sampen:.3f}\nCP: {cp:.3f}\nA: {A}\nB: {B}")
            except IndexError as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

        if self.ref_file_path:
            try:
                ref_time_series = pd.read_csv(self.ref_file_path).values.flatten()
                print(f"Loaded reference time series of length {len(ref_time_series)}")
                ref_sampen, _, ref_cp, ref_A, ref_B = calculate_sampen(ref_time_series, m, r, delay)
                self.result_label.config(text=self.result_label.cget("text") +
                                         f"\nReference SampEn for m={m}, r={r:.3f}, Delay={delay}\nSampEn: {ref_sampen:.3f}\nCP: {ref_cp:.3f}\nA: {ref_A}\nB: {ref_B}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load or calculate reference CSV file: {e}")

    def calculate_permen(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        try:
            time_series = pd.read_csv(self.file_path).values.flatten()
            print(f"Loaded time series of length {len(time_series)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            return

        m = self.m_value.get()
        delay = self.delay_value.get()

        if len(time_series) < m * delay + 1:
            messagebox.showerror("Error", "Time series is too short for the given template length and delay.")
            return

        try:
            permen = calculate_permutation_entropy(time_series, m, delay)
            result_text = f"PermEn for m={m}, Delay={delay}: {permen:.3f}"
            self.result_label.config(text=result_text)
        except IndexError as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

        if self.ref_file_path:
            try:
                ref_time_series = pd.read_csv(self.ref_file_path).values.flatten()
                ref_permen = calculate_permutation_entropy(ref_time_series, m, delay)
                result_text += f"\nReference PermEn for m={m}, Delay={delay}: {ref_permen:.3f}"
                self.result_label.config(text=result_text)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load or calculate reference CSV file: {e}")

    def plot_se_vs_r(self, r_values, se_sampen_values, optimal_r):
        plt.figure(figsize=(10, 6))
        plt.plot(r_values, se_sampen_values, marker='o')
        plt.axvline(optimal_r, color='r', linestyle='--', label=f'Optimal r: {optimal_r:.3f}')
        plt.xlabel('r values')
        plt.ylabel('Standard Error of SampEn')
        plt.title(f'Standard Error of SampEn vs r values for m={self.m_value.get()}')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = SampEnGUI(root)
    root.mainloop()
