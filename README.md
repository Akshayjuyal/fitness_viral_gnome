# fitness_viral_gnome

Here’s a structured `README.md` file with added context to describe each step:  

```markdown
# Fitness Vector Generator

## Overview

This script is designed for **preprocessing and generating fitness vectors** from genomic datasets. The tool provides different modes for processing input data and extracting fitness vectors based on SNP (Single Nucleotide Polymorphism) interactions.

## Usage

The script supports **two main operations**:
1. **Preprocessing** - Preparing the dataset for further analysis.
2. **Fitness Vector Computation** - Generating fitness vectors using statistical methods.

### **1. Preprocessing Data**
Preprocessing is required to clean and structure the dataset before fitness vector computation.

#### **Command**
```sh
python fitness_vector.py -p -in usa_matrix.csv -d usa_dates.txt -o .
```
#### **Parameters**
- `-p`, `--preprocess` → Enables the preprocessing step.
- `-in`, `--input` → Path to the **input dataset** (CSV file).
- `-d`, `--date-path` → Path to the **date file** associated with the dataset.
- `-o`, `--output-path` → Directory to save the preprocessed data (default: `.`).

---

### **2. Generating Fitness Vectors**
Once preprocessing is complete, you can compute the fitness vectors using different statistical approaches.

#### **Example 1: Basic Fitness Vector Calculation**
```sh
python fitness_vector.py -in complete_USA_sorted_with_dates.pkl -df 2020-04-06 -pv 0.5 -sn 5 -o .
```
#### **Example 2: Advanced Fitness Vector Calculation**
```sh
python fitness_vector.py -in complete_USA_sorted_with_dates.pkl -df 2020-04-06 -fd 2020-01-17 -th 0.05 -sn 5 -ct ratio -o output/
```
#### **Parameters**
- `-in`, `--input` → Path to the input dataset (CSV or `.pkl` file).
- `-fd`, `--start-date` → Start date for selecting sequences (format: `YYYY-MM-DD`).
- `-df`, `--date-filter` → End date for selecting sequences (format: `YYYY-MM-DD`).
- `-th`, `--p-value` → Threshold for statistical significance of edge calculations (**default: 0.05**).
- `-sn`, `--shuffle-times` → Number of **shuffling iterations** for randomization (**default: 5**).
- `-ct`, `--calculation-type` → Method for computing SNP interactions (**default: "ratio"**). Possible values:
  - `ratio` → Computes relative frequency changes.
  - `diff` → Computes absolute differences.
  - `binomial` → Uses binomial probability for significance.
- `-sf`, `--save-files` → Save the computed results to files (**default: False**).

---

## **Execution Flow**

1. **Preprocessing (`-p`)**  
   - Validates the **input file** and **date file**.
   - Saves cleaned data to the specified output directory.

2. **Fitness Vector Computation (without `-p`)**  
   - Reads the input dataset and extracts sequences within the given date range.
   - Computes **significant edges** using the chosen `calculation-type` method.
   - Saves the output to the specified directory.

---

## **Example Workflow**
1. **Preprocess the dataset**  
   ```sh
   python fitness_vector.py -p -in usa_matrix.csv -d usa_dates.txt -o preprocessed_data/
   ```
2. **Compute fitness vectors after preprocessing**  
   ```sh
   python fitness_vector.py -in preprocessed_data/complete_USA_sorted_with_dates.pkl -df 2020-04-06 -th 0.05 -sn 5 -ct binomial -o results/
   ```

---

## **Output Files**
- Processed data is stored in the **output directory** specified by `-o`.
- A log file (`statistics_<date>.log`) is created, summarizing:
  - Input parameters
  - Computation type used
  - Number of shuffles performed
  - Statistical thresholds applied

---

## **Error Handling**
- If **preprocessing is enabled**, both `--date-path` and `--input` must be provided.
- If **fitness vector computation is selected**, `--input` and `--date-filter` are mandatory.
- The script ensures that existing directories for a given `date-filter` are cleared before writing new results.

---

## **License**
This project is open-source and available for use under the **MIT License**.

---

## **Contributors**
- Maintained by: **Akshay Juyal**
- Contact: [akshayjuyal@gmail.com](mailto:akshayjuyal@gmail.com)

```