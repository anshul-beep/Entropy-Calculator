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

    for i in range(N - m):
        for j in range(i + 1, N - m):
            if is_match(i, j, m, time_series, r):
                B += 1
                B_pairs.append((i, j))
                if is_match(i, j, m + 1, time_series, r):
                    A += 1
                    A_pairs.append((i, j))

    if B == 0:
        return np.nan, np.nan

    CP = A / B
    SampEn = -np.log(CP)

    K_A = count_overlaps(np.array(A_pairs), m + 1)
    K_B = count_overlaps(np.array(B_pairs), m)

    SE_CP = np.sqrt((CP * (1 - CP) / B) + (1 / B**2) * (K_A + K_B * CP**2))
    SE_SampEn = SE_CP / CP

    return SampEn, SE_SampEn

@jit(nopython=True)
def find_optimal_r(time_series, m, r_values):
    se_sampen_values = np.empty(len(r_values))
    for idx in range(len(r_values)):
        r = r_values[idx]
        _, se_sampen = calculate_sampen(time_series, m, r)
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
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            return

        if self.find_optimal_r_value.get():
            r_values = np.linspace(0.01, 0.5, 50)
            optimal_r, se_sampen_values = find_optimal_r(time_series, m, r_values)
            sampen, se_sampen = calculate_sampen(time_series, m, optimal_r)
            self.result_label.config(text=f"Optimal r for m={m}: {optimal_r:.3f}\nSampEn: {sampen:.3f}")
            self.plot_se_vs_r(r_values, se_sampen_values, optimal_r)
        else:
            r = self.r_value.get()
            sampen, se_sampen = calculate_sampen(time_series, m, r)
            if np.isnan(sampen):
                messagebox.showerror("Error", "Failed to calculate Sample Entropy.")
            else:
                self.result_label.config(text=f"SampEn for m={m}, r={r:.3f}: {sampen:.3f}")

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


