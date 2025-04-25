import os
from typing import Optional, List, Dict, Union, Any
import json
import re
from pathlib import Path
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    AnnSearchRequest,
    WeightedRanker
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_openai import OpenAIEmbeddings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MilvusVectorStore")

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self, artifacts_path: str = "/home/rajeshgupta/.cache/docling/models"):
        self.heading_pattern = re.compile(r'^(#+|\d+\.\s+|•\s+)(.*)$', re.MULTILINE)
        self.artifacts_path = os.path.expanduser(artifacts_path)
        self.setup_document_converter()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    def setup_document_converter(self):
        """Initialize Docling with optimized PDF processing"""
        pipeline_options = PdfPipelineOptions(artifacts_path=self.artifacts_path)
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["en"]
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=os.cpu_count(), 
            device=AcceleratorDevice.AUTO
        )
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            },
        )
        self.chunker = HybridChunker(tokenizer="jinaai/jina-embeddings-v3", max_tokens=1000)


    def extract_chunk_metadata(self, chunk) -> Dict[str, Any]:
        """Extract metadata from Docling chunks"""
        metadata = {
            "text": chunk.text,
            "headings": [],
            "page_info": None,
        }

        if hasattr(chunk, "meta"):
            if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                metadata["headings"] = chunk.meta.headings

            if hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, "page_no"):
                                metadata["page_info"] = prov.page_no

        return metadata
    
    def extract_heading(self, text: str) -> Optional[str]:
        """Improved for Twilio's document structure"""
        # First check for Twilio's bold section headers
        bold_headers = re.findall(r'\n\*\*(.*?)\*\*\n', text)
        if bold_headers:
            return bold_headers[0]
        
        # Then check for standard markdown headers
        md_headers = re.findall(r'^#+\s+(.*)$', text, re.MULTILINE)
        if md_headers:
            return md_headers[0]
        
        # Finally look for underlined sections
        underlined = re.findall(r'\n(.*?)\n[-=]+\n', text)
        if underlined:
            return underlined[0]
        
        return None

    def process_document(self, file_path: str) -> List[Dict]:
        """Process a document (PDF or text) using Docling pipeline"""
        try:
            logger.info(f"Processing document: {file_path}")
            processed_chunks = []
            # Determine input format
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.pdf':
                result = self.converter.convert(file_path)  # Changed here: positional argument
                doc = result.document
                # Process with Docling
                chunks = list(self.chunker.chunk(doc))
                for i, chunk in enumerate(chunks):
                    metadata = self.extract_chunk_metadata(chunk)
                    metadata.update({
                        "source": str(file_path),
                        "chunk_id": i
                    })
                    
                    processed_chunks.append(metadata)
                                
            else:
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                for i, chunk in enumerate(chunks):
                    headings = self.extract_heading(chunk.page_content)
                    metadata = {
                        "text": chunk.page_content,
                        "headings": headings if headings else " ",
                        "source": str(file_path),
                        "page_info": chunk.metadata.get("page", -1),
                        "chunk_id": i,
                    }
                    processed_chunks.append(metadata)       

            
            logger.info(f"Processed {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []


    def process_files(self, file_paths: Union[str, List[str]]) -> List[Dict]:
        """Process multiple files"""
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        all_chunks = []
        for path in file_paths:
            path = str(path)  # Ensure string for Path operations
            if not Path(path).exists():
                logger.warning(f"File not found: {path}")
                continue
                
            if Path(path).is_dir():
                # Process all files in directory
                for file in Path(path).rglob("*"):
                    if file.is_file():
                        chunks = self.process_document(str(file))
                        all_chunks.extend(chunks)
            else:
                chunks = self.process_document(path)
                all_chunks.extend(chunks)
        
        return all_chunks


class MilvusVectorStore:
    def __init__(
        self,
        collection_name: str = "esrsDocuments",
        embedding_model: str = "text-embedding-3-small",
        milvus_uri: str = "documents.db",
        reset_collection: bool = False,
        load_local=True,
    ):
        self.collection_name = collection_name
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.load_local=load_local
        
        # Connect to Milvus
        connections.connect("default", uri=milvus_uri)
        
        # Initialize or reset collection
        if reset_collection and utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"Reset existing collection: {collection_name}")
        
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Create the Milvus collection if it doesn't exist"""
        if self.load_local and utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(
                f"Loaded existing collection: {self.collection_name} "
                f"with {self.collection.num_entities} entities"
            )
        else:
            self._create_collection()

    def _create_collection(self):        
        # Define schema for new collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._get_embedding_dim(),
            ),
            FieldSchema(
                name="heading_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._get_embedding_dim(),
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="headings", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="page_info", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Document chunks with embeddings for ESG gap analysis",
        )
        
        self.collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
        self.collection.create_index("embedding", index_params)
        self.collection.create_index("heading_embedding", index_params)
        self.collection.load()
        logger.info(f"Created new collection: {self.collection_name}")
    
    def _get_embedding_dim(self) -> int:
        """Get the dimensionality of the embedding model"""
        # Use a dummy text to determine embedding dimension
        dummy_embed = self.embedding_model.embed_query("test")
        return len(dummy_embed)
    
    def add_documents(self, file_paths: Union[str, List[str]]) -> int:
        """Add documents to the vector store"""
        processor = DocumentProcessor()
        chunks = processor.process_files(file_paths)
        
        if not chunks:
            logger.warning("No valid chunks found in the provided files")
            return 0
        
        # Generate embeddings in batches
        texts = [chunk["text"] for chunk in chunks]
        headings = [json.dumps(chunk["headings"]) if chunk["headings"] else "[]" for chunk in chunks]
        text_embeddings = self.embedding_model.embed_documents(texts)
        heading_embeddings = self.embedding_model.embed_documents(headings)
        
        # Prepare entities for insertion - must match schema field order
        entities = [
            text_embeddings,  # embedding field
            heading_embeddings,  # heading_embedding field
            texts,  # text field
            headings,  # heading field
            [chunk["source"] for chunk in chunks],  # source field
            [chunk["page_info"] for chunk in chunks],  # page field
            [chunk["chunk_id"] for chunk in chunks],  # chunk_id field
        ]
        
        # Insert into Milvus
        insert_result = self.collection.insert(entities)
        self.collection.flush()
        
        num_inserted = len(insert_result.primary_keys)
        logger.info(f"Inserted {num_inserted} document chunks")
        return num_inserted
    
    def search(self, query: str, k: int = 5,use_hybrid: bool = True,heading_weight: float = 0.3,content_weight: float = 0.7) -> List[Dict]:
        """Enhanced search with hybrid heading/content matching"""
        
        # Generate embeddings for both content and headings
        query_embedding = self.embedding_model.embed_query(query)
        
        if use_hybrid:
            # Hybrid search combining content and headings
            content_request = AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=k*2,
            )
            
            heading_request = AnnSearchRequest(
                data=[query_embedding],
                anns_field="heading_embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=k*2,
            )
            
            results = self.collection.hybrid_search(
                reqs=[content_request, heading_request],
                rerank=WeightedRanker(content_weight, heading_weight),
                limit=k,
                output_fields=["embedding","text", "headings", "source", "page_info"],
            )[0]
        else:
            # Standard semantic search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=k,
                output_fields=["embedding","text", "headings", "source", "page_info"],
            )[0]
        
        formatted_results = []
        for hit in results:
            formatted_results.append({
                "embedding":hit.entity.get("embedding"),
                "text": hit.entity.get("text"),
                "headings": hit.entity.get("headings"),
                "source": hit.entity.get("source"),
                "page_info": hit.entity.get("page_info"),
                "score": hit.score,
            })
        
        return formatted_results
        
    def clear_collection(self):
        """Delete all documents in the collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        else:
            logger.warning(f"Collection {self.collection_name} does not exist")


"""# Example usage
if __name__ == "__main__":
    # Initialize vector store
    vector_store = MilvusVectorStore(
        collection_name="publicDocuments",
        embedding_model="text-embedding-3-small",
        milvus_uri="docling_vector_store.db",
        load_local=True,
        reset_collection=False
    )
    
    # Add documents (can be a file or directory)
    documents_dir = "./new documents"  # Path to your documents
    vector_store.add_documents(documents_dir)
    
    # Example search
    query = "What are Twilio's policies on human rights?"
    results = vector_store.search(query, k=3)
    
    print(f"\nSearch results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page_info']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Content:\n{result['text']}\n")"""