import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model (you can change to any other SBERT model if needed)
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and lightweight

def process_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Compute similarity score between ground truth and recommendation per row."""
    scores = []

    for _, row in df.iterrows():
        ground_truth = str(row.get("Ground_Truth_Anthesis_Recommendations", "")).strip()
        recommendation = str(row.get("Recommendations", "")).strip()

        if ground_truth and recommendation:
            embeddings = model.encode([ground_truth, recommendation], convert_to_tensor=True)
            score = float(util.cos_sim(embeddings[0], embeddings[1]))
        else:
            score = 0.0

        scores.append(score)

    df["Recommendation_Score"] = scores
    return df

def process_excel(file_path: str, output_path: str):
    """Processes all sheets and saves with new similarity column."""
    xls = pd.ExcelFile(file_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            processed_df = process_sheet(df)
            processed_df.to_excel(writer, sheet_name=sheet_name, index=False)

# Example usage
input_file ="/home/rajeshgupta/Downloads/esrs_extraction_results_all_sheets_ESRS_including_websearch_modified_query.xlsx"
output_file = "/home/rajeshgupta/Downloads/esrs_extraction_results_all_sheets_ESRS_including_websearch_modified_query_rec_score.xlsx"
process_excel(input_file, output_file)
