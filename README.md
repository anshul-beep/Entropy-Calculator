# Entropy Calculator

A Python-based GUI application for calculating various types of entropy measures from time series data. This application supports Sample Entropy, Permutation Entropy, Approximate Entropy, and Spectral Entropy calculations with customizable parameters and windowing options.

## Features

- **Multiple Entropy Types**:
  - Sample Entropy
  - Permutation Entropy
  - Approximate Entropy
  - Spectral Entropy

- **Advanced Windowing Options**:
  - Normal windowing with custom number of windows
  - Sliding window with overlap percentage
  - Hamming window application option

- **Flexible Parameter Selection**:
  - Manual or automatic template length (m) selection
  - Multiple options for tolerance (r) value selection:
    - Manual input
    - Automatic selection
    - Common r value for all windows
    - Window-specific optimization

- **Data Processing**:
  - Support for CSV and Excel files
  - Batch processing of multiple files
  - Efficient handling of large datasets through chunking

- **Visualization and Export**:
  - Real-time plotting of entropy results
  - Export results to CSV with detailed window data
  - Interactive GUI with modern theming

## Requirements

```
python >= 3.6
tkinter
ttkthemes
PIL
numpy
pandas
matplotlib
statsmodels
EntropyHub
scipy
```

## Installation

1. Clone the repository or download the source code
2. Install the required packages:
```bash
pip install tkinter ttkthemes pillow numpy pandas matplotlib statsmodels entropyhub scipy
```

## Usage

1. Run the application:
```bash
python entropy_calculator.py
```

2. Using the GUI:
   - Click "Load Files" to select your input files (CSV or Excel)
   - Choose the entropy type and windowing method
   - Set your desired parameters
   - Click "Calculate" to process the data
   - Use "Plot" to visualize results
   - Click "Download" to save results as CSV

## Parameter Guide

### General Parameters
- **Template Length (m)**: The length of compared runs of data
- **Time Delay**: The delay between data points in template matching
- **Window Size**: The number of data points in each analysis window
- **Overlap Percentage**: The percentage of overlap between consecutive windows

### Entropy-Specific Parameters

#### Sample/Approximate Entropy
- **Tolerance (r)**: The similarity criterion
  - Manual: User-specified value
  - Auto: Automatically selected based on data
  - Common: Same value for all windows
  - Optimized: Individual optimization per window

#### Permutation Entropy
- **Permutation Type**: Different methods of permutation analysis
  - uniquant
  - finegrain
  - modified
  - ampaware
  - weighted
  - edge
  - phase

#### Spectral Entropy
- **FFT Length (N)**: Length of the Fast Fourier Transform
- **Frequency Range**: Min and max frequencies for analysis

## Output Format

The CSV output includes:
- File path
- Window number
- Entropy value
- Normalized entropy (where applicable)
- Conditional entropy (where applicable)
- r value (where applicable)
- Window data

## Tips for Best Results

1. **Data Preparation**:
   - Ensure your time series data is clean and properly formatted
   - Remove any NaN or infinite values
   - Consider normalizing your data if comparing across different scales

2. **Parameter Selection**:
   - Start with standard values (m=2 or 3, r=0.2*std)
   - Use auto-selection features for optimal parameters
   - Consider the length of your time series when selecting window sizes

3. **Performance Optimization**:
   - Use appropriate chunk sizes for large datasets
   - Balance window size and overlap percentage for meaningful results
   - Consider using normal windowing for very large datasets

## Troubleshooting

Common issues and solutions:

1. **File Loading Errors**:
   - Ensure files are in CSV or Excel format
   - Check file permissions
   - Verify data format consistency

2. **Calculation Errors**:
   - Check if time series length is sufficient for selected parameters
   - Verify data contains no invalid values
   - Ensure window size is appropriate for the data length

3. **Performance Issues**:
   - Reduce chunk size for large files
   - Decrease window overlap percentage
   - Use normal windowing instead of sliding window for very large datasets

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
