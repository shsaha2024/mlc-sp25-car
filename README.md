# A Cashflow-Based Machine Learning Model for Scoring Credit-Invisible Consumers

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-enabled-orange.svg)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
We develop a cashflow-based credit scoring model to assess default risk for 'credit invisible' consumers who lack traditional credit histories. Using transaction data from checking and savings accounts, we extract features that capture spending patterns, volatility, and temporal trends. In addition to standard aggregations, we apply spectral analysis via a Fast Fourier Transform (FFT) to uncover periodic financial behavior. Our models, trained per loan product using XGBoost, demonstrate reasonable accuracy and offer a promising approach for inclusive credit risk assessment.

## Problem Statement
Traditional credit scores (like FICO) rely heavily on historical credit data, excluding approximately 70 million "credit invisible" or "thin-file" consumers in the US. This project develops a cashflow-based credit scoring model using only transaction-level and account-level data. By framing the task as a binary classification problem, we predict loan defaults using historical transaction data leading up to the evaluation date.

## Dataset Structure
The dataset spans four different lending institutions (`C01`, `C02`, `C03`, `C04`), each offering distinct financial products (personal loans, payday loans, credit cards). The data is structured at two levels:

**Consumer-Level Data:**
- Masked Consumer ID
- Total account balance on the evaluation date
- Evaluation date
- Binary default indicator (`FPF_TARGET`)

**Transaction-Level Data:**
- Posted date
- Transaction amount (positive for credits, negative for debits)
- Categorical label for transaction type
- Masked Consumer ID

> **Note:** To prevent data leakage, only transactions occurring *before* the evaluation date were used for model training and inference.

## 🛠️ Methodology & Feature Engineering
Our pipeline was systematically developed for each client, focusing heavily on robust feature engineering to quantify financial stability, spending habits, and income volatility.

### Key Feature Categories
1. **Aggregate Transaction Statistics:** Basic health indicators like total amount, mean amount, and credit/debit ratios.
2. **Temporal and Volatility Features:** Monthly cash flow volatility, 30-day activity, and linear regression-based monthly spending trends.
3. **Category-Specific Features:** Net spending, usage flags, and average time between transactions across 36 distinct categories (e.g., rent, food, healthcare).
4. **Weekly Dynamic Features:** Short-term patterns using 3-week rolling windows, weekly changes, and weekend spending ratios.
5. **Spectral Analysis (FFT):** Employed specifically for `C03` to uncover latent periodic structures (dominant frequency, spectral entropy, high/low-frequency power) reflecting rhythmic cash flow behaviors and financial volatility.

### Modeling Approach
We adopted an iterative modeling strategy focusing on gradient boosting architectures:
- **Primary Algorithm:** XGBoost Classifier (chosen for handling non-linearities, missing values, and imbalanced data).
- **Baselines Tested:** Logistic Regression (with SMOTE/Platt scaling) and Decision Trees.
- **Hyperparameter Tuning:** Conducted via `RandomizedSearchCV` with 3-fold and 5-fold cross-validation, optimizing for maximum Validation AUC while applying regularization (L1/L2) to combat overfitting.
- **Class Imbalance Handling:** Utilized `scale_pos_weight` to address the low default rate (e.g., ~3.5% to 4.8% depending on the client).

## Key Results
Model performance was evaluated using the minimum Area Under the ROC Curve (AUC) across the client segments to promote fairness.

| Client | Best Model | Validation AUC | Test AUC | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **C01** | XGBoost (All 521 Features) | **0.7621** | - | Comprehensive feature set outperformed reduced sets. |
| **C02** | XGBoost (Top 144 Features)| **0.8177** | - | Feature selection significantly improved performance. |
| **C03** | XGBoost (Top 125 Features)| **0.7650** | 0.805 | Heavily benefited from Spectral Analysis (FFT) features. |
| **C04** | XGBoost (Full Feature Set) | **0.7600** | 0.623 | Experienced distributional shift on test data. |

## Retrospective & Future Work
- **What Went Right:** Translating raw transaction logs into custom-engineered features (temporal, categorical, and spectral) was the most critical success factor. XGBoost consistently outperformed linear models.
- **Areas for Improvement:** 
  - Overfitting remained a challenge, particularly in `C03` and `C04`. Stricter regularization (capping tree depth, increasing `min_child_weight`) is necessary.
  - Future iterations must address temporal distributional shifts (observed in `C04`) by employing out-of-time validation sets and resampling techniques to account for changing macroeconomic conditions.

## Contributors
- **Joey Kaminsky**
- **Jason Chen**
- **Manat Rao**
- **Shubham Saha**
- **Zhaolong Han**

*University of California, San Diego (UC San Diego)*
