import requests
from langchain_community.document_loaders import SeleniumURLLoader
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from bs4 import BeautifulSoup
import os
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean and normalize the extracted text."""
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common unhelpful patterns
    patterns_to_remove = [
        r'Skip to main content',
        r'!function\(.*?\){.*?}\(.*?\);?',
        r'<!--.*?-->',
        r'<noscript>.*?</noscript>',
        r'<iframe.*?</iframe>',
        r'<script.*?</script>',
        r'<style.*?</style>',
        r'Loading\.\.\.',
        r'window\._[\w]+=window\._[\w]+\|\|\[\];',
        r'googletagmanager\.com/ns\.html'
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL|re.IGNORECASE)
    
    return text

def extract_main_content(soup):
    """Extract main content sections from the page."""
    # Try to find main content areas - these are specific to Twilio's structure
    main_content = soup.find('main', {'id': 'main-content'})
    if not main_content:
        main_content = soup.find('div', class_='body__inner')
    
    if main_content:
        # Remove unwanted elements from main content
        for element in main_content.find_all(['nav', 'footer', 'aside', 'header', 'form', 'iframe', 'div', 'section']):
            if 'nav' in element.get('class', []) or 'footer' in element.get('class', []):
                element.decompose()
        
        return str(main_content)
    return str(soup)

def load_html_content(urls: list) -> list:
    # Set USER_AGENT to avoid warnings
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Automatically download and manage chromedriver
    service = Service(ChromeDriverManager().install())

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        documents = []
        for url in urls:
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(["script", "style", "meta", "link", "noscript", "iframe", "svg", "img"]):
                element.decompose()
            
            # Extract main content area
            html_content = extract_main_content(soup)
            clean_soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get text and clean it
            text = clean_soup.get_text()
            text = clean_text(text)
            
            from langchain_core.documents import Document
            documents.append(Document(page_content=text, metadata={"source": url}))
        
        return documents
    finally:
        driver.quit()

def _ensure_unique_filename(path: str) -> str:
    """Avoid overwriting existing files by adding a number if needed."""
    path = Path(path)
    if not path.exists():
        return str(path)
    
    counter = 1
    while True:
        new_path = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not new_path.exists():
            return str(new_path)
        counter += 1

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
        urls=list(set(urls))
        print(len(urls))

    for url in urls:
        try:
            documents = load_html_content([url])
            content = "\n\n".join([doc.page_content for doc in documents])

            if "twilio.com/" in url:
                name_part = url.split("twilio.com/")[1]
            else:
                name_part = url.split("https://")[1].replace('/','').replace('.','_')

            file_name = "Twilio-" + name_part.replace("/", "-").title() + ".txt"
            file_path = os.path.join(output_dir, file_name)
            file_path = _ensure_unique_filename(file_path)
            with open(file_path, "w", encoding="utf-8") as f_out:
                f_out.write(content)

            print(f"[✓] Saved: {file_path}")
        except Exception as e:
            print(f"[!] Failed to process {url}: {e}")

# Example usage
if __name__ == "__main__":
    input_txt_file = "/home/rajeshgupta/Downloads/downloaded_links_firefox.txt"
    output_directory = "./new documents"
    process_urls_from_file(input_txt_file, output_directory)