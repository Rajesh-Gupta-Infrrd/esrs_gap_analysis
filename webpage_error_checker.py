import os

def check_error_in_txt_files(directory_path):
    """
    Scans all .txt files in the given directory and checks
    if they contain 'ERROR:' or 'Access Denied'.
    Prints matching file names and the total count.
    """
    if not os.path.isdir(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    error_count = 0
    deny_count=0
    flag=False
    deleted=0
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if "ERROR:" in content :
                        flag=True
                        print(f"Match found in: {filename}")
                        error_count += 1
                    elif "Access Denied" in content:
                        print(f"Match found in: {filename}")
                        flag=True
                        deny_count += 1
            except Exception:
                # Skipping files that can't be read
                continue

    print(f"\nTotal files with 'ERROR:'{error_count} or 'Access Denied': {deny_count} and Deleted {deleted}")

# Example usage
if __name__ == "__main__":
    dir_path = "/home/rajeshgupta/Git_Clones/esrs_gap_analysis/new documents"
    check_error_in_txt_files(dir_path)
