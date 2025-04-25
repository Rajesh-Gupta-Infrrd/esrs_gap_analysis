import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from vector_store import MilvusVectorStore
import logging
import requests
from bs4 import BeautifulSoup
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ESGProcessor")

class ESGExcelProcessor:
    def __init__(self, public_vector_store: MilvusVectorStore, esrs_vector_store: MilvusVectorStore):
        self.public_vector_store = public_vector_store
        self.esrs_vector_store = esrs_vector_store
        self.url = "https://xbrl.efrag.org/e-esrs/esrs-set1-2023.html"

    def get_text_from_specific_section(self, url: str, section_id: str) -> Optional[str]:
        """Retrieve text from a specific section of a webpage with error handling."""
        if not url or pd.isna(url) or str(url).lower() == 'nan':
            return None
            
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            section = soup.find(id=section_id)
            
            return section.get_text(separator='\n', strip=True) if section else None
        except Exception as e:
            logger.warning(f"Failed to retrieve content from {url}: {str(e)}")
            return None

    def split_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Splits a URL with optional section ID with robust error handling."""
        if not url or pd.isna(url) or str(url).lower() == 'nan':
            return None, None
        
        try:
            parts = str(url).split(" - ")
            if len(parts) == 1:
                return parts[0].strip(), None
            return parts[0].strip(), parts[-1].strip()
        except Exception as e:
            logger.warning(f"Failed to split URL {url}: {str(e)}")
            return None, None
    
    def process_excel(self, input_path: str, output_path: str, top_k: int = 3) -> Optional[str]:
        """Process the ESG Excel sheet and create enriched output with improved error handling."""
        try:
            all_sheets = pd.read_excel(input_path, sheet_name=None)
            logger.info(f"Loaded Excel file with sheets: {list(all_sheets.keys())}")
            
            enriched_sheets = {}
            
            for sheet_name, df in all_sheets.items():
                logger.info(f"Processing sheet: {sheet_name} with {len(df)} rows")
            
                required_cols = ['Question', 'Disclosure Requirement', 'Hyperlink']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Missing required columns in sheet {sheet_name}")
                    continue
                
                results = []
                for _, row in df.iterrows():
                    try:
                        question = str(row['Question'])
                        requirements = str(row['Disclosure Requirement'])
                        hyperlink = row['Hyperlink']
                        Ground_Truth_Readiness=row['Readiness']
                        Ground_Truth_Overview_of_GAP=row['Overview of GAP']
                        Ground_Truth_Anthesis_Recommendations=row['Anthesis Recommendations']
                        
                        # Handle hyperlink processing
                        url, section_id = self.split_url(hyperlink)
                        query_text = None
                        
                        if url and section_id:
                            query_text = self.get_text_from_specific_section(url, section_id)
                        
                        # Fallback to requirements if hyperlink processing failed
                        query_text = query_text or requirements
                        
                        # Search based on disclosure requirements
                        esrs_results = []
                        esrs_score = 0.0
                        if query_text:
                            esrs_results = self.esrs_vector_store.search(query_text, k=1)
                            esrs_score = sum(res['score'] for res in esrs_results)/1 if esrs_results else 0.0
                        
                        esrs_content = self._combine_results(esrs_results)
                        
                        combined_query = f"{question}\n{esrs_content}"
                        
                        # Search based on question
                        document_results = self.public_vector_store.search(combined_query, k=top_k)
                        document_score = sum(res['score'] for res in document_results)/top_k if document_results else 0.0
                        document_content = self._combine_results(document_results)
                        
                        results.append({
                            'question': question,
                            'requirements': requirements,
                            'Public Document Content': document_content,
                            'document_score': document_score,
                            'ESRS Document Content': esrs_content,
                            'requirements_score': esrs_score,
                            'Original Hyperlink': hyperlink,
                            'Ground_Truth_Readiness':Ground_Truth_Readiness,
                            'Ground_Truth_Overview_of_GAP':Ground_Truth_Overview_of_GAP,
                            'Ground_Truth_Anthesis_Recommendations':Ground_Truth_Anthesis_Recommendations

                        })
                    except Exception as row_error:
                        logger.error(f"Error processing row {_}: {str(row_error)}")
                        continue
                    
                    enriched_sheets[sheet_name] = pd.DataFrame(results)
            
            if not enriched_sheets:
                logger.error("No sheets were successfully processed")
                return None
            
            # Save to new Excel file
            output_path = self._ensure_unique_filename(output_path)
            with pd.ExcelWriter(output_path) as writer:
                for sheet_name, df in enriched_sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            logger.info(f"Successfully saved enriched results to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}", exc_info=True)
            return None

    def _combine_results(self, search_results: List[Dict]) -> str:
        """Combine search results into a single text with sources."""
        if not search_results:
            return ""
        return "\n".join(res['text'] for res in search_results)

    def _ensure_unique_filename(self, path: str) -> str:
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


# Example Usage
if __name__ == "__main__":
    # Initialize Milvus vector store
    public_vector_store = MilvusVectorStore(
        collection_name="publicDocuments",
        embedding_model="text-embedding-3-small",
        milvus_uri="docling_vector_store.db",
        load_local=True
    )
    esrs_vector_store=MilvusVectorStore(
        collection_name="esrsDocuments",
        embedding_model="text-embedding-3-small",
        milvus_uri="docling_vector_store.db",
        load_local=True
    )
    # Create processor
    processor = ESGExcelProcessor(public_vector_store,esrs_vector_store)
    
    # Process Excel file
    input_excel = "Download_ESRS GAP_Twilio (1).xlsx"  # Your input file
    output_excel = "esrs_extraction_results_all_files_sheets_hyperlink_top1.xlsx"  # Output will be created
    
    try:
        result_path = processor.process_excel(input_excel, output_excel)
        print(f"Processing complete. Results saved to: {result_path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")