import json
import os
from google import genai
from dotenv import load_dotenv



load_dotenv()
def build_evidence(df, training_results=None):
    evidence = {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": df.columns.tolist(),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": {
                column: int(count)
                for column, count in df.isna().sum().items()
            },
        }
    }

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        evidence["numeric_summary"] = json.loads(
            numeric_df.describe().to_json()
        )

    if training_results is not None:
        target = training_results["target_column"]

        if target in df.columns:
            evidence["target_counts"] = {
                str(value): int(count)
                for value, count in df[target].value_counts().items()
            }

        evidence["model"] = {
            "selected_model": training_results["best_model_name"],
            "positive_class": "Yes",
            "train_rows": int(training_results["train_rows"]),
            "test_rows": int(training_results["test_rows"]),
            "test_metrics": {
                metric: float(training_results[metric])
                for metric in ["accuracy", "precision", "recall", "f1"]
            },
            "baseline_accuracy": float(
                training_results["baseline_accuracy"]
            ),
            "baseline_f1": float(training_results["baseline_f1"]),
            "feature_importance": training_results["feature_importance"],
        }

    return evidence
def ask_copilot(question, evidence):
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")

    if not api_key or not model_name:
        raise ValueError(
            "Set GEMINI_API_KEY and GEMINI_MODEL in your .env file."
        )

    instructions = (
        "You are a data-analysis assistant. Answer only from the supplied "
        "computed evidence. Do not invent statistics or imply you saw raw "
        "customer rows. If evidence is insufficient, say so. Distinguish "
        "prediction from causation. Keep answers concise."
    )

    prompt = json.dumps({
        "question": question,
        "computed_evidence": evidence,
    })

    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=instructions,
                max_output_tokens=700,
            ),
        )

    return response.text