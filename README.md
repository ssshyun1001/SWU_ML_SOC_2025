## 🔋 Machine Learning-Based SOC Prediction for Lithium-ion Batteries


### 🏆 Award
This research was selected as an **Excellent Paper (Student Division)** at the **2025 Industrial Safety Research Competition**,  
hosted by the Industrial Safety Mutual Foundation and the Korean Society of Safety. <br>
📅 Date: September 17, 2025

<br>

### 📌 Research Background
Accurate State of Charge (SOC) estimation is critical for preventing overcharge and over-discharge. This is because it occurs fire accident and explosion in lithium-ion batteries.
Traditional SOC estimation methods such as Open Circuit Voltage (OCV), Coulomb Counting, Impedance Track suffer from cumulative error.
Deep learning models provide high accuracy but suffer from High computational cost and Low interpretability.
This study proposes an interpretable and computationally efficient machine learning-based SOC prediction model.

<br>

### 📊 Dataset

Source: AI Hub – High-quality R&D Lithium-ion Battery Dataset

Chemistry: NCM (Nickel-Cobalt-Manganese)
Raw samples: 9,844,440

Final processed samples: 8,324,355
Final features: 74

<br>

### ⚙️ Data Preprocessing
- One-hot encoding for categorical variables
- Atomic feature extraction using Pymatgen
- Min-max scaling
- Feature selection: Correlation < 0.95 , VIF < 10, P-value < 0.05

<br>

### 🧠 Models Compared

- Linear Regression
- Decision Tree Regression
- XGBoost Regression (final model)

Hyperparameter tuning performed using Optuna.

<br>

### 📈 Model Performance
| Model | Train MSE | Test MSE | Test R² |
|--------|------------|------------|------------|
| Linear Regression | 20.800 | 20.855 | 0.976 |
| Decision Tree | 23.903 | 23.928 | 0.973 |
| XGBoost | 0.041 | 0.056 | 1.000 |

XGBoost achieved the lowest error and highest explanatory power.

<br>

### 🔎 Explainability (SHAP Analysis)

To enhance interpretability, SHAP (SHapley Additive exPlanations) was applied.

Top influencing variables:

- tm_o_bond_length
- chemical_ordering

  <br>

**Key insights**

Shorter TM-O bond length → Higher SOC prediction
Lower structural ordering → Higher SOC
> This confirms the physical relevance of structural parameters in SOC behavior.

<br>

### 🎯 Industrial Significance
- Enables explainable SOC prediction
- Supports Battery Management System (BMS) optimization
- Enhances industrial safety by preventing battery failures
- Balances accuracy, efficiency, and interpretability

<br>

### 🏆 Research Context
2025 Industrial Safety Research Competition
