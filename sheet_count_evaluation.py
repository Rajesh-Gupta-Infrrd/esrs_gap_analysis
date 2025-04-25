import pandas as pd

# File paths
ground_file = "/home/rajeshgupta/Git_Clones/esrs_gap_analysis/Download_ESRS GAP_Twilio (1).xlsx"
results_file = "/home/rajeshgupta/Downloads/esrs_extraction_results_all_sheets_ESRS_including_websearch_modified_query.xlsx"


# Load all sheets
ground_sheets = pd.read_excel(ground_file, sheet_name=None)
results_sheets = pd.read_excel(results_file, sheet_name=None)

# Comparison summary
for sheet_name in ground_sheets.keys():
    

    df_ground = ground_sheets[sheet_name]
    df_results = results_sheets[sheet_name]

    # Ensure both sheets have the 'Readiness' column
    if "Readiness" not in df_ground.columns or "Readiness" not in df_results.columns:
        print(f"Missing 'Readiness' column in sheet '{sheet_name}'.")

    # Align DataFrames by index (assumes rows correspond 1-to-1)
    min_len = min(len(df_ground), len(df_results))
    df_ground = df_ground.iloc[:min_len]
    df_results = df_results.iloc[:min_len]

    correct = 0
    yes=0
    no=0
    partial=0
    wrong = 0
    wrong_combinations = {}

    for i in range(min_len):
        val_ground = str(df_ground.iloc[i]["Readiness"]).strip().lower()
        val_result = str(df_results.iloc[i]["Readiness"]).strip().lower()
        if val_ground == val_result:
            correct += 1
        else:
            wrong += 1
        if val_ground == 'yes' and val_result == 'yes':
            yes+=1
        elif val_ground == 'partially' and val_result == 'partially':
            partial+=1
        elif val_ground == 'no' and val_result == 'no':
            no+=1
        else:
            # Build a combination key like "yes->partial"
            key = f"{val_ground}->{val_result}"
            if key not in wrong_combinations:
                wrong_combinations[key] = {'count': 0, 'indices': []}
            wrong_combinations[key]['count'] += 1
            wrong_combinations[key]['indices'].append(i+1)

    #Output correct counts
    print("Correct counts:")
    print(f"YES: {yes}")
    print(f"PARTIAL: {partial}")
    print(f"NO: {no}")

    # Output all wrong combinations with indices
    print("\nWrong combinations and their indices:")
    for combo, data in wrong_combinations.items():
        print(f"{combo}: {data['count']} times at indices {data['indices']}")#

    print(f"Sheet: {sheet_name} | Correct: {correct} | Wrong: {wrong}")
