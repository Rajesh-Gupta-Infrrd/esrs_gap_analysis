import pdfkit
import os
def convert_url_to_pdf(url, pdf_path):
    """
    Converts a website URL to a PDF file.

    Args:
        url: The URL of the website.
        pdf_path: The desired path for the output PDF file.
    """
    try:
        pdfkit.from_url(url, pdf_path)
        print(f"PDF generated and saved at {pdf_path}")
    except Exception as e:
        print(f"PDF generation failed: {e}")


def process_urls_from_file(txt_path: str, output_dir: str):
    """
    Read URLs from a .txt file and save each page's content to a file.

    Args:
        txt_path (str): Path to the .txt file containing URLs (one per line).
        output_dir (str): Directory to save extracted content files.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(txt_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        try:

            if "twilio.com/" in url:
                name_part = url.split("twilio.com/")[1]
            else:
                name_part = "unknown"

            file_name = "Twilio-" + name_part.replace("/", "-").title() + ".pdf"
            file_path = os.path.join(output_dir, file_name)
            convert_url_to_pdf(url, file_path)
    
            print(f"[✓] Saved: {file_path}")
        except Exception as e:
            print(f"[!] Failed to process {url}: {e}")

# Example usage
if __name__ == "__main__":
    input_txt_file = "/home/rajeshgupta/Downloads/downloaded_links_firefox.txt"
    output_directory = "./pdfs"
    process_urls_from_file(input_txt_file, output_directory)
