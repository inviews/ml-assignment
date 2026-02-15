import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
)


# Configuration

st.set_page_config(
    page_title="Online Shoppers Intention Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model", "model")
METRICS_DIR = os.path.join(BASE_DIR, "model", "metrics")
DATA_DIR = os.path.join(BASE_DIR, "model", "data")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")
TEST_SET_PATH = os.path.join(DATA_DIR, "test_set.csv")


MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

METRICS_FILES = {
    "Logistic Regression": "logistic_regression.json",
    "Decision Tree": "decision_tree.json",
    "KNN": "knn.json",
    "Naive Bayes": "naive_bayes.json",
    "Random Forest": "random_forest.json",
    "XGBoost": "xgboost.json",
}


# Helper Functions

@st.cache_resource
def load_resources():

    resources = {}
    

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            resources["scaler"] = pickle.load(f)
    else:
        st.error(f"Scaler not found at {SCALER_PATH}")
        return None

    if os.path.exists(X_TRAIN_PATH):
        df_cols = pd.read_csv(X_TRAIN_PATH, nrows=0)
        resources["expected_cols"] = df_cols.columns.tolist()
    else:
        st.error(f"Training data header not found at {X_TRAIN_PATH}")
        return None
        
    return resources

def load_model(model_name):

    filename = MODELS.get(model_name)
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model file not found: {path}")
        return None

def load_all_metrics():

    results = []
    for name, filename in METRICS_FILES.items():
        path = os.path.join(METRICS_DIR, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                metrics = json.load(f)
                metrics["model"] = name 
                results.append(metrics)
    
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"model": "Model"}, inplace=True)

    # Select and rename for display
    display_cols = {
        "Model": "ML Model Name",
        "accuracy": "Accuracy",
        "auc": "AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "mcc": "MCC"
    }
    final_df = df.rename(columns=display_cols)
    return final_df[[c for c in display_cols.values() if c in final_df.columns]]

@st.cache_data
def load_test_set_bytes():

    with open(TEST_SET_PATH, "rb") as f:
        return f.read()

def preprocess_input(df, expected_cols, scaler):

    df = df.copy()
    y_true = None
    
    # Check for target variable
    if "Revenue" in df.columns:

        if df["Revenue"].dtype == 'bool' or df["Revenue"].dtype == 'object':
             df["Revenue"] = df["Revenue"].astype(int)
        y_true = df["Revenue"]
        df = df.drop(columns=["Revenue"])
    

    if set(df.columns) == set(expected_cols):
        X = df[expected_cols]
    else:

        if "Weekend" in df.columns:
             df["Weekend"] = df["Weekend"].astype(int)
        

        categorical_cols = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']

        existing_cats = [c for c in categorical_cols if c in df.columns]
        
        if existing_cats:
            df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
        

        X = df.reindex(columns=expected_cols, fill_value=0)

    # Scale
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        return None, None, f"Scaling error: {str(e)}"

    return X_scaled, y_true, None


# Main App Layout

def main():
    st.title("Online Shoppers Intention Prediction")
    st.markdown("""
    This application predicts whether a visitor will make a purchase (Revenue) based on their browsing behavior.
    It demonstrates the performance of multiple Machine Learning models trained on the **Online Shoppers Purchasing Intention Dataset**.
    """)

    # Sidebar — My info
    st.sidebar.markdown("""
    <div style="background-color:#ff6b00;padding:14px 16px;border-radius:8px;margin-bottom:16px;">
        <p style="color:#ffffff;font-weight:700;font-size:1rem;margin:0;">ML Assignment 2</p>
        <p style="color:#ffffff;font-weight:600;font-size:0.95rem;margin:4px 0 0 0;">Sumanth Reddy</p>
        <p style="color:#ffffff;font-weight:600;font-size:0.95rem;margin:4px 0 0 0;">2025AA05795</p>
        <a style="color:#ffffff;font-weight:600;font-size:0.95rem;margin:4px 0 0 0;" href="https://github.com/inviews/ml-assignment" target="_blank">https://github.com/inviews/ml-assignment</a>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar — download sample test data
    st.sidebar.header("Sample Test Data")
    st.sidebar.markdown("Download the preprocessed test set to try the **Predict & Evaluate** page.")
    if os.path.exists(TEST_SET_PATH):
        test_bytes = load_test_set_bytes()
        st.sidebar.download_button(
            label="⬇️ Download test_set.csv",
            data=test_bytes,
            file_name="test_set.csv",
            mime="text/csv",
        )
    else:
        st.sidebar.warning("test_set.csv not found.")

    # TODO: Replace the URL below with the actual hosted link to test_set.csv
    TEST_SET_URL = "https://github.com/inviews/ml-assignment/raw/main/model/data/test_set.csv"
    st.sidebar.markdown(
        f'📎 You can download from  [this link]({TEST_SET_URL})  if button failed',
        unsafe_allow_html=False,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pages**")
    st.sidebar.markdown("Use the tabs above to switch between pages.")


    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 54px;
        padding: 0 28px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        background-color: #1e3a5f;
        color: #a0b8d8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e6da4 !important;
        color: #ffffff !important;
        border-bottom: 3px solid #4da3ff !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #264d73;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    resources = load_resources()
    if not resources:
        st.stop()

    # Tab-based navigation — Predict & Evaluate first
    tab1, tab2, tab3, tab4 = st.tabs(["Predict & Evaluate", "Model Comparison", "Metric Visualization", "Visualization"])

    
    # Tab 2: Model Comparison
    
    with tab2:
        st.header("Model Performance Metrics")
        
        metrics_df = load_all_metrics()
        
        if not metrics_df.empty:
            st.dataframe(metrics_df.style.highlight_max(subset=metrics_df.columns[1:], axis=0, color='lightgreen'), use_container_width=True, height=280)
            
            st.subheader("Analysis")
            st.markdown("""
            *The table above highlights the best performing model for each metric.*
            *Check "Metric Visualization" & "Visualization" for more insights*
            """)
            
        else:
            st.warning("No metrics found. Please train the models first.")

    
    # Tab 3: Metric Visualization
    
    with tab3:
        st.header("Metric Visualization")
        metrics_df = load_all_metrics()
        if not metrics_df.empty:
            for metric in metrics_df.columns[1:]:
                st.subheader(metric)
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(
                    data=metrics_df,
                    x=metric, y="ML Model Name",
                    hue="ML Model Name", ax=ax,
                    palette="viridis", legend=True,
                    orient="h",
                )
                ax.set_title("")
                ax.set_ylabel("")
                ax.set_xlim(0, 1.08)
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.4f", padding=3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.divider()
        else:
            st.warning("No metrics found. Please train the models first.")

    
    # Tab 4: Visualization
    
    with tab4:
        st.header("Visualization")
        metrics_df = load_all_metrics()
        if not metrics_df.empty:
            metric_cols = list(metrics_df.columns[1:])
            models = metrics_df["ML Model Name"].tolist()

            # Heatmap
            st.subheader("Metrics Heatmap")
            st.caption("Color intensity shows relative performance — darker = higher score.")
            heatmap_data = metrics_df.set_index("ML Model Name")[metric_cols]
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(
                heatmap_data, annot=True, fmt=".4f",
                cmap="YlGn", vmin=0, vmax=1,
                linewidths=0.5, linecolor="white", ax=ax,
            )
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.divider()

            # Radar / Spider chart
            st.subheader("Radar Chart — Model Performance Profile")
            st.caption("Each spoke is a metric. A wider shape means stronger overall performance.")
            N = len(metric_cols)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1] 

            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(models)))

            for i, model in enumerate(models):
                vals = metrics_df.loc[metrics_df["ML Model Name"] == model, metric_cols].values.flatten().tolist()
                vals += vals[:1]
                ax.plot(angles, vals, "o-", linewidth=2, color=colors[i], label=model)
                ax.fill(angles, vals, alpha=0.08, color=colors[i])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_cols, size=11)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color="grey")
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.divider()

            # Rank Heatmap 
            st.subheader("Model Rank Heatmap")
            st.caption("Rank 1 = best for that metric. Green = top performer, Red = lowest.")
            rank_data = heatmap_data.rank(ascending=False).astype(int)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(
                rank_data, annot=True, fmt="d",
                cmap="RdYlGn_r", vmin=1, vmax=len(models),
                linewidths=0.5, linecolor="white", ax=ax,
            )
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.divider()

        else:
            st.warning("No metrics found. Please train the models first.")

    
    # Tab 1: Predict & Evaluate
    
    with tab1:
        st.header("Prediction & Evaluation")
        
        # 1. Upload Data
        st.subheader("1. Upload Test Data (CSV)")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            _preview_df = pd.read_csv(uploaded_file)
            st.write(f"**Uploaded Data Shape:** {_preview_df.shape}")
            with st.expander("Preview Uploaded Data"):
                st.dataframe(_preview_df.head())
            uploaded_file.seek(0)

        # 2. Select Model
        st.subheader("2. Select Model")
        st.markdown("""
        <style>
        div[data-testid="stRadio"] label {
            font-size: 1.05rem;
            font-weight: 700;
            color: #000000;
        }
        div[data-testid="stRadio"] > div {
            background-color: #1e3a5f;
            border: 2px solid #2e6da4;
            border-radius: 10px;
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            width: 100%;
        }
        div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
            font-size: 0.95rem;
            font-weight: 600;
            color: #d0e8ff;
        }
        </style>
        """, unsafe_allow_html=True)
        selected_model_name = st.radio("Choose a model:", list(MODELS.keys()), horizontal=True)
        
        if uploaded_file is not None:
            # Load Data
            try:
                df = pd.read_csv(uploaded_file)

                # Preprocess
                X_scaled, y_true, error = preprocess_input(df, resources["expected_cols"], resources["scaler"])
                
                if error:
                    st.error(error)
                else:
                    # Predict
                    model = load_model(selected_model_name)
                    if model:
                        if st.button("▶ Run Prediction", type="primary", use_container_width=True):
                            with st.spinner("Predicting..."):
                                y_pred = model.predict(X_scaled)
                                # Try predict_proba
                                try:
                                    y_prob = model.predict_proba(X_scaled)[:, 1]
                                except Exception:
                                    y_prob = None
                                

                                if y_true is not None:
                                    st.markdown("""
                                    <div style="background-color:#1e3a5f;padding:16px 20px;border-radius:8px;margin-bottom:12px;">
                                        <h2 style="color:#ffffff;margin:0;">📝 Evaluation on Uploaded Data</h2>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # All 6 required metrics
                                    m_acc = accuracy_score(y_true, y_pred)
                                    m_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
                                    m_prec = precision_score(y_true, y_pred, zero_division=0)
                                    m_rec = recall_score(y_true, y_pred, zero_division=0)
                                    m_f1 = f1_score(y_true, y_pred, zero_division=0)
                                    m_mcc = matthews_corrcoef(y_true, y_pred)

                                    mc1, mc2, mc3 = st.columns(3)
                                    mc1.metric("Accuracy", f"{m_acc:.4f}")
                                    mc2.metric("AUC", f"{m_auc:.4f}" if m_auc is not None else "N/A")
                                    mc3.metric("Precision", f"{m_prec:.4f}")
                                    mc4, mc5, mc6 = st.columns(3)
                                    mc4.metric("Recall", f"{m_rec:.4f}")
                                    mc5.metric("F1 Score", f"{m_f1:.4f}")
                                    mc6.metric("MCC", f"{m_mcc:.4f}")

                                    st.markdown("""
                                    <div style="background-color:#1e3a5f;padding:12px 20px;border-radius:8px;margin-top:16px;margin-bottom:12px;">
                                        <h3 style="color:#ffffff;margin:0;">🔲 Confusion Matrix & Classification Report</h3>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        cm = confusion_matrix(y_true, y_pred)
                                        fig_cm, ax_cm = plt.subplots()
                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                        ax_cm.set_xlabel('Predicted')
                                        ax_cm.set_ylabel('Actual')
                                        st.pyplot(fig_cm)

                                    with col2:
                                        report = classification_report(y_true, y_pred, output_dict=True)
                                        st.dataframe(pd.DataFrame(report).transpose())


                                st.divider()
                                st.subheader("Prediction Results")

                                results_df = df.copy()
                                if "Revenue" in results_df.columns:
                                    results_df = results_df.drop(columns=["Revenue"])
                                results_df["Predicted_Revenue"] = y_pred
                                if y_prob is not None:
                                    results_df["Probability"] = y_prob

                                st.dataframe(results_df.head(10))

                                # Download
                                csv = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download Predictions",
                                    csv,
                                    "predictions.csv",
                                    "text/csv",
                                    key='download-csv'
                                )

            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("Please upload a CSV file to proceed. It can be raw data or processed data. If 'Revenue' column is present, evaluation metrics will be shown.")

if __name__ == "__main__":
    main()
