# Machine Learning 

# ML Assignment 2

*Sumanth Reddy P*
*2025AA05795*
*[2025aa05795@wilp.bits-pilani.ac.in](mailto:2025aa05795@wilp.bits-pilani.ac.in)*


# Links

### Github

Repo : [https://github.com/inviews/ml-assignment](https://github.com/inviews/ml-assignment)

### Streamlit

Streamlit : [https://ml-assignment-2025aa05795.streamlit.app/](https://ml-assignment-2025aa05795.streamlit.app/)



# A - Problem Statement

**Predict whether an online shopping session will result in a purchase (revenue generation).**

The task is a **binary classification** problem: given a set of behavioural and session-level features collected from an e-commerce website, predict whether the user's visit ends in a completed transaction (`Revenue = True`) or not (`Revenue = False`).

This has direct business value, identifying high-intent users allows a website to offer targeted promotions, prioritise support, or personalise the experience in real time to maximise conversion rates.

---

# B - Dataset Description

[https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

| Property               | Detail                                                       |
| ---------------------- | ------------------------------------------------------------ |
| **Name**               | Online Shoppers Purchasing Intention Dataset                 |
| **Source**             | UCI Machine Learning Repository                              |
| **Total Instances**    | 12330                                                        |
| **Features (raw)**     | 18 columns (10 numerical, 8 categorical)                     |
| **Target Variable**    | `Revenue` Binary (False = no purchase, True = purchase)      |
| **Class Distribution** | ~84.5% Non-purchase (Class 0), ~15.5% Purchase (Class 1)     |
| **Train / Test Split** | 80% train (~9,856 rows) / 20% test (~2,474 rows), stratified |

### Features

**Numerical:**

|Feature|Description|
|---|---|
|`Administrative`|Number of pages visited of administrative type|
|`Administrative_Duration`|Total time (seconds) spent on administrative pages|
|`Informational`|Number of informational pages visited|
|`Informational_Duration`|Total time on informational pages|
|`ProductRelated`|Number of product-related pages visited|
|`ProductRelated_Duration`|Total time on product-related pages|
|`BounceRates`|Avg. bounce rate of pages visited (from Google Analytics)|
|`ExitRates`|Avg. exit rate of pages visited|
|`PageValues`|Avg. page value of pages visited before completing a transaction|
|`SpecialDay`|Closeness of the visit date to a special day (e.g., Valentine's Day)|

**Categorical:**

| Feature            | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `Month`            | Month of the visit (Jan–Dec)                                  |
| `OperatingSystems` | OS used (integer-coded nominal)                               |
| `Browser`          | Browser used (integer-coded nominal)                          |
| `Region`           | Geographic region (integer-coded nominal)                     |
| `TrafficType`      | Traffic source type (integer-coded nominal)                   |
| `VisitorType`      | Returning Visitor, New Visitor, or Other                      |
| `Weekend`          | Boolean, was the session on a weekend                         |
| `Revenue`          | **Target**: Boolean, did the session result in a transaction? |

### Preprocessing Applied

1. **Missing Values:** handelled missing values
2. **Duplicate Removal:** Duplicate rows removed.
3. **Encoding:**
    - One-Hot Encoding
    - Boolean Encoding for binary fields
4. **Scaling:** `StandardScaler` fitted on training set only, then applied to both train and test to prevent data leakage.
5. **Stratified split** preserves the ~84.5/15.5% class ratio in both sets.

---

# C - Models

### Comparison Table - Evaluation Metrics

| **ML Model Name**       | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
| ----------------------- | ------------ | ------- | ------------- | ---------- | ------ | ------- |
| **Logistic Regression** | 0.8878       | 0.8981  | 0.7547        | 0.4188     | 0.5387 | 0.5078  |
| **Decision Tree**       | 0.8615       | 0.7484  | 0.5547        | 0.5838     | 0.5689 | 0.4867  |
| **kNN**                 | 0.8578       | 0.7075  | 0.6277        | 0.2251     | 0.3314 | 0.3163  |
| **Naive Bayes**         | 0.2450       | 0.5813  | 0.1693        | 0.9791     | 0.2887 | 0.1088  |
| **Random Forest**       | 0.9013       | 0.9218  | 0.7509        | 0.5524     | 0.6365 | 0.5901  |
| **XGBoost**             | 0.8964       | 0.9246  | 0.6903        | 0.6126     | 0.6491 | 0.5900  |



# Observation

### Logistic Regression

Logistic Regression delivers a **strong and reliable baseline** for this binary classification problem. With an accuracy of **88.78%** and an AUC of **0.8981**, it is the best-performing linear model. Logistic Regression is a good choice when **minimising false positives**. But linear nature also means it cannot capture the complex, non-linear interactions between features (`PageValues`, `BounceRates`, and `ExitRates`), that tree-based models can. The performance issues is likely due to precision-recall trade-off in an imbalanced dataset. The performance is average compared to other models.

---

### Decision Tree

The Decision Tree achieves an accuracy of **86.15%**  but shows a notably lower AUC of **0.7484** compared to logistic regression and the ensemble methods. This is likely due to a overfit tree. The Decision Tree has **better Recall (0.5838) than Logistic Regression (0.4188)**, meaning it identifies more actual purchasers. Decision Tree lags behind models like random Forest and Naive Bayes. The Decision Tree is useful for interpretability , but for production use on this dataset, ensemble models are better.

---

### KNN

K-Nearest Neighbor (KNN) also did not perform well. While the accuracy is 85%, taking class imbalanec into account, this is not a good score. The **F1 score of 0.3314** and **MCC of 0.3163** are the lowest among all models. This is due to two reasons

- Higher Dimensions: One Hot Encoded data has many dimensions and so distance between features became useless. This contradicts, KNN's core mechanism of "similarity by proximity."
- Data Imbalance : We need to SMOTE oversample the data to use properly.

KNN is also not a very good model for this data.

---

### Naive Bayes

Naive Bayes Performed poorly on almost all metrics. The root cause is likely that Naive Bayes assumes that all features follow a Gaussian (normal) distribution, but this was not the case. The dataset contains only one-hot encoded binary featurse.  **Recall is 0.9791 (98%)** while **Precision is only 0.1693 (17%)**. This pattern conclusively indicates the model is predicting **Class 1 (Revenue = True)** for almost every single instance in the test set. This is not a good data for Naive Bayes and should be feature engineered a lot if model performance should be improved.

---

### Random Forest

Random Forest is the **best model by Accuracy and MCC** and similar to XGBoost in AUC. With an accuracy of **90.13%**, AUC of **0.9218**, Precision of **0.7509**, and MCC of **0.5901**, it demonstrates strong, well-rounded performance across all metrics. Random Forest's superiority over the single Decision Tree is dramatic. Random Forest is particularly suited to this dataset because it naturally handles **mixed data types** (the dataset has both continuous metrics and one-hot encoded values), is robust to outliers, and implicitly captures **non-linear feature interactions**. It is close to XGBoost in many metrics and can be a good fit for deployment.

---

### XGBoost

XGBoost achieves the **best AUC (0.9246), best Recall (0.6126), and best F1 score (0.6491)** among all six models, making it the **overall best-performing model** for this purchase-prediction task. Its accuracy of **89.64%** is slightly below Random Forest (90.13%). The **best-in-class AUC of 0.9246** means XGBoost has the best ability to rank and discriminate between purchasing and non-purchasing sessions, this is the most theretically important metric for a real-world deployment. XGBoost's superior recall and F1 stem from its **gradient boosting** mechanism: unlike Random Forest (which builds trees independently in parallel), XGBoost builds trees **sequentially**. XGBoost also incorporates **L1/L2 regularisation** (unlike standard decision trees), which controls overfitting while retaining the ability to model complex non-linear relationships. if goal is maximum revenue  generation, XGBoost should be used.

---


### Key Observations : 

**Ensemble methods performs well**: Random Forest and XGBoost outperform all other models on every meaningful metric.

**Class imbalance** : Since class is imbalance, we should not just look at accuracy.

**Feature Engineering**: More Feature Engineering is needed based on each model. This is avoided to keep same data for all models as mentioned in Assignment Question.
