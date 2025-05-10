---

# 💧 VFP Optimization App

This Streamlit app provides an interactive interface for **Vertical Flow Performance (VFP) optimization**, helping reservoir engineers and analysts determine the optimal **Tubing Head Pressure (THP)** and **flow rate** to minimize drawdown or achieve a target Bottom Hole Pressure (BHP) for CO₂ injection wells.

---

## 📊 Features

- Upload your own VFP data as a CSV (columns: `THP`, `FlowRate`, `BHP`)
- Visualize and explore your VFP curve interactively
- Set target BHP or drawdown pressure constraints
- Optimize for THP and FlowRate using different methods:
  - Nelder-Mead (derivative-free)
  - Powell
  - L-BFGS-B (with bounds)
  - COBYLA (constrained optimization)
- Compare methods and download optimized results

---

## 📂 File Upload

The app expects a CSV file with the following format:

```csv
THP,FlowRate,BHP
1000,10.0,4000
1100,12.5,4200
1200,15.0,4400
...
````

If no file is uploaded, the app will use a **placeholder dataset** included in this repo.

---

## 🛠 How to Run Locally

1. **Clone the repository**:

   ```bash
   git clone https://github.com/okpoEkpenyong/vfp_optimization_app.git
   cd vfp_optimization_app
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   streamlit run vfp_model_optimization.py
   ```

---

## 🚀 Deploy via Streamlit Cloud

You can deploy this app in one click by linking your GitHub repo to [Streamlit Cloud](https://streamlit.io/cloud).

---

## 📁 Project Structure

```
vfp-optimization-app/
│
├── vfp_model_optimization.py                 # Streamlit application
├── input.csv   # Example VFP data
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 📬 Contact

For questions, improvements, or feedback, feel free to open an issue or reach out.

---

