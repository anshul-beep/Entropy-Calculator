import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

@jit(nopython=True)
def is_match(x1, x2, length, time_series, r):
    return np.max(np.abs(time_series[x1:x1+length] - time_series[x2:x2+length])) < r

@jit(nopython=True)
def calculate_pairwise_distances(time_series, length):
    N = len(time_series)
    distances = np.zeros((N - length, N - length))
    for i in range(N - length):
        for j in range(i + 1, N - length):
            distances[i, j] = np.max(np.abs(time_series[i:i + length] - time_series[j:j + length]))
            distances[j, i] = distances[i, j]
    return distances

@jit(nopython=True)
def count_overlaps(pairs, length):
    overlap_count = 0
    for (i, j) in pairs:
        for (k, l) in pairs:
            if (i, j) == (k, l):
                continue
            if min(abs(i - k), abs(i - l), abs(j - k), abs(j - l)) <= length:
                overlap_count += 1
    return overlap_count

@jit(nopython=True)
def calculate_sampen(time_series, m, r):
    N = len(time_series)
    B = 0
    A = 0
    B_pairs = []
    A_pairs = []

    distances_m = calculate_pairwise_distances(time_series, m)
    distances_m1 = calculate_pairwise_distances(time_series, m + 1)

    for i in range(N - m):
        for j in range(i + 1, N - m):
            if i != j and distances_m[i, j] < r:
                B += 1
                B_pairs.append((i, j))
                if distances_m1[i, j] < r:
                    A += 1
                    A_pairs.append((i, j))

    if B == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, distances_m, distances_m1, B_pairs, A_pairs

    CP = A / B
    SampEn = -np.log(CP)

    K_A = count_overlaps(np.array(A_pairs), m + 1)
    K_B = count_overlaps(np.array(B_pairs), m)

    SE_CP = np.sqrt((CP * (1 - CP) / B) + (1 / B**2) * (K_A + K_B * CP**2))
    SE_SampEn = SE_CP / CP

    return SampEn, SE_SampEn, CP, A, B, distances_m, distances_m1, B_pairs, A_pairs

@jit(nopython=True)
def find_optimal_r(time_series, m, r_values):
    se_sampen_values = np.empty(len(r_values))
    for idx in range(len(r_values)):
        r = r_values[idx]
        _, se_sampen, _, _, _, _, _, _, _ = calculate_sampen(time_series, m, r)
        se_sampen_values[idx] = se_sampen if not np.isnan(se_sampen) else np.inf

    optimal_r_index = np.argmin(se_sampen_values)
    optimal_r = r_values[optimal_r_index]

    return optimal_r, se_sampen_values

class SampEnGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sample Entropy Calculator")

        self.file_path = ""
        self.m_value = tk.IntVar(value=2)
        self.r_value = tk.DoubleVar(value=0.2)
        self.find_optimal_r_value = tk.BooleanVar()

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Sample Entropy Calculator", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(frame, text="Template Length (m):").grid(row=1, column=0, sticky=tk.W)
        self.m_entry = ttk.Entry(frame, textvariable=self.m_value)
        self.m_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))

        ttk.Label(frame, text="Tolerance (r):").grid(row=2, column=0, sticky=tk.W)
        self.r_entry = ttk.Entry(frame, textvariable=self.r_value)
        self.r_entry.grid(row=2, column=1, sticky=(tk.W, tk.E))

        self.optimal_r_check = ttk.Checkbutton(frame, text="Find optimal r", variable=self.find_optimal_r_value)
        self.optimal_r_check.grid(row=3, column=0, columnspan=2)

        ttk.Button(frame, text="Load CSV", command=self.load_csv).grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Calculate", command=self.calculate).grid(row=5, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(frame, text="")
        self.result_label.grid(row=6, column=0, columnspan=2, pady=10)

        for child in frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path = file_path
            messagebox.showinfo("File Loaded", f"Loaded file: {file_path}")

    def calculate(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        m = self.m_value.get()
        try:
            time_series = pd.read_csv(self.file_path).values.flatten()
            print(f"Loaded time series of length {len(time_series)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            return

        if len(time_series) < m + 1:
            messagebox.showerror("Error", "Time series is too short for the given template length.")
            return

        if self.find_optimal_r_value.get():
            std_dev = np.std(time_series)
            r_min = 0.1 * std_dev
            r_max = 0.25 * std_dev
            r_values = np.linspace(r_min, r_max, 50)
            try:
                optimal_r, se_sampen_values = find_optimal_r(time_series, m, r_values)
                sampen, se_sampen, cp, A, B, distances_m, distances_m1, B_pairs, A_pairs = calculate_sampen(time_series, m, optimal_r)
                self.result_label.config(text=f"Optimal r for m={m}: {optimal_r:.3f}\nSampEn: {sampen:.3f}\nCP: {cp:.3f}\nA: {A}\nB: {B}")
                self.plot_se_vs_r(r_values, se_sampen_values, optimal_r)
                self.plot_distance_matrices(distances_m, distances_m1, m, optimal_r, B_pairs, A_pairs)
            except IndexError as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
        else:
            r = self.r_value.get()
            try:
                sampen, se_sampen, cp, A, B, distances_m, distances_m1, B_pairs, A_pairs = calculate_sampen(time_series, m, r)
                if np.isnan(sampen):
                    messagebox.showerror("Error", "Failed to calculate Sample Entropy.")
                else:
                    self.result_label.config(text=f"SampEn for m={m}, r={r:.3f}: {sampen:.3f}\nCP: {cp:.3f}\nA: {A}\nB: {B}")
                    self.plot_distance_matrices(distances_m, distances_m1, m, r, B_pairs, A_pairs)
            except IndexError as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

    def plot_se_vs_r(self, r_values, se_sampen_values, optimal_r):
        plt.figure(figsize=(10, 6))
        plt.plot(r_values, se_sampen_values, marker='o')
        plt.axvline(optimal_r, color='r', linestyle='--', label=f'Optimal r: {optimal_r:.3f}')
        plt.xlabel('r values')
        plt.ylabel('Standard Error of SampEn')
        plt.title(f'Standard Error of SampEn vs r values for m={self.m_value.get()}')
        plt.legend()
        plt.show()

    def plot_distance_matrices(self, distances_m, distances_m1, m, r, B_pairs, A_pairs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Pairwise Distance Matrices for m={m} and m={m+1}', fontsize=16)

        cax0 = axes[0].matshow(distances_m, cmap='viridis')
        axes[0].set_title(f'Distances for m={m}')
        fig.colorbar(cax0, ax=axes[0])

        cax1 = axes[1].matshow(distances_m1, cmap='viridis')
        axes[1].set_title(f'Distances for m={m+1}')
        fig.colorbar(cax1, ax=axes[1])

        # Highlight matched pairs
        for (i, j) in B_pairs:
            if distances_m[i, j] < r:
                axes[0].plot([j], [i], marker='o', color='red')
                axes[0].plot([i], [j], marker='o', color='red')

        for (i, j) in A_pairs:
            if distances_m1[i, j] < r:
                axes[1].plot([j], [i], marker='o', color='red')
                axes[1].plot([i], [j], marker='o', color='red')

        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = SampEnGUI(root)
    root.mainloop()

