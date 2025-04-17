import pandas as pd
from pathlib import Path
from typing import List, Dict
from vector_store import MilvusVectorStore  # Assuming your class is in this module
import logging

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
    
    def process_excel(self, input_path: str, output_path: str, top_k: int = 3):
        """
        Process the ESG Excel sheet and create enriched output
        
        Args:
            input_path: Path to input Excel file
            output_path: Path to save enriched Excel file
            top_k: Number of search results to retrieve per query
        """
        try:
            all_sheets = pd.read_excel(input_path, sheet_name=None)
            logger.info(f"Loaded Excel file with sheets: {list(all_sheets.keys())}")
            
            enriched_sheets = {}
            
            for sheet_name, df in all_sheets.items():
                # Read input Excel
                logger.info(f"Processing sheet: {sheet_name} with {len(df)} rows")
            
                # Validate required columns
                required_cols = ['Question', 'Disclosure Requirement']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Input Excel must contain columns: {required_cols}")
                
                # Process each question
                results = []
                for _, row in df.iterrows():
                    question = str(row['Question'])
                    requirements = str(row['Disclosure Requirement'])
                    
                    # Search based on question
                    document_results = self.public_vector_store.search(question, k=top_k)
                    document_content = self._combine_results(document_results)
                    document_score = sum ([res['score'] for res in document_results])/top_k
                    
                    # Search based on disclosure requirements
                    esrs_results = self.esrs_vector_store.search(requirements, k=top_k)
                    esrs_content = self._combine_results(esrs_results)
                    esrs_score = sum ([res['score'] for res in esrs_results])/top_k
                    
                    results.append({
                        'question': question,
                        'requirements': requirements,
                        'Pubic Document Content': document_content,
                        'document_score' : document_score,
                        'ESRS Document Content': esrs_content,
                        'requirements_score':esrs_score

                    })
                enriched_sheets[sheet_name] = pd.DataFrame(results)
            
            if not enriched_sheets:
                raise ValueError("No sheets were processed due to missing required columns.")
            
            # Save to new Excel file
            output_path = self._ensure_unique_filename(output_path)
            # Write all enriched sheets to Excel
            with pd.ExcelWriter(output_path) as writer:
                for sheet_name, df in enriched_sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            logger.info(f"Saved enriched results to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")


    def _combine_results(self, search_results: List[Dict]) -> str:
        """Combine search results into a single text with sources"""
        combined = []
        for i, res in enumerate(search_results, 1):
            combined.append(
                f"{res['text']}\n"
            )
        return "\n".join(combined)

    def _ensure_unique_filename(self, path: str) -> str:
        """Avoid overwriting existing files by adding a number if needed"""
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
    output_excel = "esrs_extraction_results_all_sheets.xlsx"  # Output will be created
    
    try:
        result_path = processor.process_excel(input_excel, output_excel)
        print(f"Processing complete. Results saved to: {result_path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")