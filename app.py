import pandas as pd
import streamlit as st
import plotly.express as px
from modeling import train_churn_model, predict_new_customers
from copilot import build_evidence, ask_copilot

st.set_page_config(
    page_title="Data Copilot",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Copilot")
st.write("Clean your data, explore patterns, train models, and ask questions.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
            )
        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        st.subheader("Dataset preview")
        st.dataframe(df.head(10))

        st.subheader("Data quality report")

        duplicate_count = int(df.duplicated().sum())
        st.metric("Duplicate rows", duplicate_count)

        quality_report = pd.DataFrame({
            "Column": df.columns,
            "Data type": df.dtypes.astype(str).values,
            "Missing values": df.isna().sum().values,
            "Missing (%)": df.isna().mean().mul(100).round(2).values,
        })

        st.dataframe(quality_report, hide_index=True)


        st.caption(
            "Empty cells and whitespace-only text count as missing. "
            "Duplicate rows are exact matches after blank normalization."
        )

        st.subheader("Clean your dataset")

        remove_duplicates = st.checkbox("Remove duplicate rows")
        drop_missing = st.checkbox("Remove rows with any missing values")

        cleaned_df = df.copy()

        if remove_duplicates:
            rows_before = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            st.success(
                f"Removed {rows_before - len(cleaned_df)} duplicate rows."
            )

        if drop_missing:
            rows_before = len(cleaned_df)
            cleaned_df = cleaned_df.dropna()
            st.success(
                f"Removed {rows_before - len(cleaned_df)} rows with missing values."
            )

        st.caption(
            "Removing incomplete rows can discard useful data. "
            "Leave this unchecked to keep those rows."
        )

        st.write(f"Rows after cleaning: {len(cleaned_df)}")
        st.dataframe(cleaned_df.head(10))

        st.download_button(
            label="Download cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

        st.subheader("Exploratory Data Analysis")

        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if cleaned_df.empty:
            st.info("No rows remain. Adjust your cleaning options.")
        elif numeric_columns:
            st.write("Summary statistics for numeric columns")
            st.dataframe(cleaned_df[numeric_columns].describe())

            selected_column = st.selectbox(
                "Choose a numeric column",
                options=numeric_columns,
            )

            numeric_fig = px.histogram(
                cleaned_df,
                x=selected_column,
                nbins=30,
                title=f"Distribution of {selected_column}",
            )

            st.plotly_chart(numeric_fig, use_container_width=True)
        else:
            st.info("No numeric columns were detected.")

        st.subheader("Category counts")

        categorical_columns = cleaned_df.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns.tolist()

        if not cleaned_df.empty and categorical_columns:
            selected_category = st.selectbox(
                "Choose a categorical column",
                options=categorical_columns,
            )

            counts = (
                cleaned_df[selected_category]
                .astype("string")
                .fillna("(Missing)")
                .value_counts()
                .head(20)
                .rename_axis("Category")
                .reset_index(name="Count")
            )

            category_fig = px.bar(
                counts,
                x="Category",
                y="Count",
                title=f"Most frequent categories in {selected_category}",
            )

            st.plotly_chart(category_fig, use_container_width=True)
            st.caption("Shows up to 20 categories, including missing values.")
        else:
            st.info("No categorical data is available to plot.")
        st.subheader("Numeric correlations")

        if len(numeric_columns) >= 2 and len(cleaned_df) >= 2:
            correlations = cleaned_df[numeric_columns].corr()

            correlation_fig = px.imshow(
                correlations,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation between numeric columns",
            )

            st.plotly_chart(
                correlation_fig,
                use_container_width=True,
            )

            st.caption(
                "Values near +1 indicate a positive linear relationship; "
                "values near -1 indicate a negative linear relationship. "
                "Correlation does not establish causation. "
                "Constant columns have undefined correlations."
            )
        else:
            st.info(
                "At least two numeric columns and two rows are needed."
            )
        st.subheader("ML Experiment Lab")

        if cleaned_df.empty:
            st.info("Upload data with rows to use the ML Lab.")
        else:
            target_column = st.selectbox(
                "Which column do you want to predict?",
                options=cleaned_df.columns.tolist(),
            )

            available_features = [
                column
                for column in cleaned_df.columns
                if column != target_column
            ]

            default_features = [
                column
                for column in available_features
                if column.lower() != "customerid"
            ]

            feature_columns = st.multiselect(
                "Select columns to use for prediction",
                options=available_features,
                default=default_features,
            )

            st.caption(
                "Exclude customerID and columns that would only "
                "be known after churn."
            )

            target_counts = (
                cleaned_df[target_column]
                .value_counts(dropna=False)
                .rename_axis("Target value")
                .reset_index(name="Rows")
            )

            st.write("Target value counts")
            st.dataframe(target_counts, hide_index=True)

            st.caption(
                "This model requires a target with Yes and No values."
            )

            if st.button("Train and compare models"):
                with st.spinner("Training and comparing models..."):
                    results = train_churn_model(
                        cleaned_df,
                        target_column,
                        feature_columns,
                    )

                st.session_state["training_results"] = results
                st.success("Training complete!")

            if "training_results" in st.session_state:
                results = st.session_state["training_results"]

                st.subheader("Training cross-validation results")

                cv_table = pd.DataFrame(
                    results["cv_comparison"]
                ).sort_values(
                    "CV churn F1",
                    ascending=False,
                )

                st.dataframe(cv_table, hide_index=True)

                st.success(
                    f"Selected model: {results['best_model_name']}"
                )

                st.caption(
                    "Selection uses training cross-validation. "
                    "The metrics below use the held-out test set."
                )

                st.write(
                    f"Training rows: {results['train_rows']} | "
                    f"Test rows: {results['test_rows']}"
                )

                metric_columns = st.columns(4)

                metric_columns[0].metric(
                    "Accuracy",
                    f"{results['accuracy']:.1%}",
                )
                metric_columns[1].metric(
                    "Precision",
                    f"{results['precision']:.1%}",
                )
                metric_columns[2].metric(
                    "Recall",
                    f"{results['recall']:.1%}",
                )
                metric_columns[3].metric(
                    "F1 score",
                    f"{results['f1']:.3f}",
                )

                st.caption("Positive class: Yes — customer churn.")

                st.subheader("Model vs baseline")

                comparison = pd.DataFrame({
                    "Model": [
                        "Most-frequent-class baseline",
                        results["best_model_name"],
                    ],
                    "Accuracy": [
                        results["baseline_accuracy"],
                        results["accuracy"],
                    ],
                    "Churn F1": [
                        results["baseline_f1"],
                        results["f1"],
                    ],
                })

                st.dataframe(
                    comparison.style.format({
                        "Accuracy": "{:.1%}",
                        "Churn F1": "{:.3f}",
                    }),
                    hide_index=True,
                )

                st.caption(
                    "Both models use the same test rows. "
                    "Churn F1 combines precision and recall."
                )

                st.subheader("Confusion matrix")

                confusion_table = pd.DataFrame(
                    results["confusion_matrix"],
                    index=["Actual No", "Actual Yes"],
                    columns=["Predicted No", "Predicted Yes"],
                )

                st.dataframe(confusion_table)

                st.subheader("Feature importance")

                importance_df = pd.DataFrame(
                    results["feature_importance"]
                ).sort_values(
                    "Mean F1 decrease",
                    ascending=False,
                )

                st.dataframe(importance_df, hide_index=True)

                importance_fig = px.bar(
                    importance_df,
                    x="Mean F1 decrease",
                    y="Feature",
                    orientation="h",
                    error_x="Standard deviation",
                    title="Permutation feature importance",
                )

                importance_fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(
                    importance_fig,
                    use_container_width=True,
                )

                st.caption(
                    "Larger positive values mean shuffling that feature "
                    "reduced test F1 more. Importance measures predictive "
                    "usefulness and does not establish causation."
                )

        st.subheader("Gemini Data Copilot")

        question = st.text_input(
            "Ask about this dataset",
            placeholder="Are there missing values or duplicate records?",
        )

        if st.button("Ask Gemini", key="ask_gemini"):
            if not question.strip():
                st.info("Type a question first.")
            else:
                try:
                    training_results = st.session_state.get(
                        "training_results"
                    )

                    evidence = build_evidence(
                        df,
                        training_results,
                    )

                    with st.spinner(
                        "Gemini is reviewing the computed evidence..."
                    ):
                        answer = ask_copilot(
                            question.strip(),
                            evidence,
                        )

                    if answer:
                        st.markdown(answer)
                    else:
                        st.warning(
                            "Gemini returned no text. Try again later."
                        )

                except Exception as error:
                    st.error(f"Gemini could not answer: {error}")
    except Exception as error:
        st.error(f"Could not process this CSV: {error}")
st.subheader("Predict churn for new customers")

if "training_results" not in st.session_state:
    st.info("Train a model first to enable predictions.")
else:
    new_customer_file = st.file_uploader(
        "Upload a new customer CSV",
        type=["csv"],
        key="new_customer_upload",
    )

    st.caption(
        "Include the same predictor columns used during training. "
        "The Churn column is not required."
    )

    if new_customer_file is not None:
        try:
            new_customers = pd.read_csv(new_customer_file)

            predictions_df = predict_new_customers(
                new_customers,
                st.session_state["training_results"],
            )

            st.dataframe(predictions_df, hide_index=True)

            st.download_button(
                label="Download customer predictions",
                data=predictions_df.to_csv(index=False).encode("utf-8"),
                file_name="customer_predictions.csv",
                mime="text/csv",
            )

            st.caption(
                "ChurnProbability ranges from 0 to 1. "
                "These are model estimates, not guaranteed outcomes."
            )

        except Exception as error:
            st.error(f"Could not predict: {error}")