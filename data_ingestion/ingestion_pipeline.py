import os
import re
import uuid
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http import models


class DataPipeline:
    def __init__(self, pdf_path, embedding_model_name="text-embedding-3-small", chunk_size=700, chunk_overlap=200):
        load_dotenv(find_dotenv())
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY")
        os.environ["QDRANT_CLOUD_ENDPOINT"] = os.getenv("QDRANT_CLOUD_ENDPOINT")
        self.pdf_path = pdf_path
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.converter = self._init_converter()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model_name, dimensions=512)

    def _init_converter(self):
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.do_cell_matching = True
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def load_and_convert(self):
        result = self.converter.convert(self.pdf_path)
        return result.document
    
    def split_markdown_tables(self, text):
        pattern = r'((?:\|.+\|\n)+)'
        blocks = []
        last_end = 0
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if last_end < start:
                blocks.append(text[last_end:start])
            blocks.append(match.group())
            last_end = end
        if last_end < len(text):
            blocks.append(text[last_end:])
        return blocks
    
    def summarize_tables_in_markdown(self, blocks):
        table_summary_prompt = PromptTemplate(
            input_variables=["table"],
            template=(
                "You are an expert data analyst. Given the following markdown table, "
                "summarize the table without missing key insights,patterns."
                "Extract all information from each record of the table and make into an summary in english"
                "don't miss out the context from the columns, and rows in the table."
                "These summaries will be embedded and used for retrieval process."
                "Do not repeat the table, only output the summary. Table:\n\n{table}"
            ),
        )
        final_blocks = []
        for block in blocks:
            block_strip = block.strip()
            if block_strip.startswith('|') and block_strip.endswith('|'):
                prompt = table_summary_prompt.format(table=block)
                summary = self.llm.invoke(prompt).content.strip()
                final_blocks.append(summary + "\n")
            else:
                final_blocks.append(block)
        return "".join(final_blocks)
    
    def chunk_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.create_documents([text])
        print(f"Text chunking complete. Created {len(chunks)} chunks.")

        if not chunks:
            print("Chunking failed. Exiting.")
            exit()

        #remove unwanted chunks
        final_chunks = chunks[:94]

        print("After filtering (only first 94 chunks):")
        return final_chunks

    def embed_chunks(self, chunks):
        chunk_texts = [doc.page_content for doc in chunks]
        chunk_embeddings = self.embeddings.embed_documents(chunk_texts)
        for i, doc in enumerate(chunks):
            doc.metadata["embedding"] = chunk_embeddings[i]
            doc.metadata["chunk_id"] = str(uuid.uuid4())
        return chunks
    

    def ingest_to_qdrant(self,extracted_data, 
                     collection_name="document_chunks_rag", 
                     vector_dim=512):
        
        """
        Connects to Qdrant, creates a collection, and ingests embeddings.
        """
        try:
            print(f"Attempting to connect to Qdrant at ...")
            client = QdrantClient(url=os.getenv("QDRANT_CLOUD_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY"),https=True)
            info = client.get_collections()
            print("Qdrant connection successful! Collections:", [c.name for c in info.collections])
        except Exception as e:
            print(f"Qdrant connection failed: {e}")
            print("Please ensure your Qdrant cloud is running (check qdrant cluster).")
            return

        # Drop existing collection if it exists
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            print(f"Dropped existing collection '{collection_name}'")

        # Create HNSW index configuration
        hnsw_config = models.HnswConfigDiff(
            m=16,
            ef_construct=200,
            full_scan_threshold=10000
        )

        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dim,
                distance=models.Distance.COSINE
            ),
            hnsw_config=hnsw_config,
            on_disk_payload=False
        )
        print(f"Created Qdrant collection '{collection_name}' with HNSW index (COSINE, m=16, ef=200)")

        # Prepare points for upsert
        points = [
        models.PointStruct(
            id=i,
            vector=record["embedding"],
            payload={"page_content": record["page_content"]}
        )
        for i, record in enumerate(extracted_data, start=1)
        ]

        # Upsert points into the collection
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Inserted {len(points)} records into Qdrant.")


    def process(self, collection_name="document_chunks_rag", vector_dim=512):
        document = self.load_and_convert()
        markdown_text = document.export_to_markdown()
        print("PDF loaded and converted to markdown.")

        blocks = self.split_markdown_tables(markdown_text)
        print(f"Split markdown into {len(blocks)} blocks.")

        summarized_text = self.summarize_tables_in_markdown(blocks)
        print("Tables summarized.")

        chunks = self.chunk_text(summarized_text)
        print(f"Text chunked into {len(chunks)} chunks.")

        print("Embedding chunks...")
        chunks_with_embeddings = self.embed_chunks(chunks)
        print("Chunks embedded.")

        extracted_data = [
            {"page_content": doc.page_content, "embedding": doc.metadata["embedding"]}
            for doc in chunks_with_embeddings
        ]
        print(f"Extracted {len(extracted_data)} chunks with embeddings.")

        self.ingest_to_qdrant(
            extracted_data,
            collection_name=collection_name,
            vector_dim=vector_dim
        )
        print("Data ingestion to Qdrant completed.")
        print("You can now use the Qdrant collection for retrieval tasks.")
        print("Make sure your Qdrant server is running and accessible.")


if __name__ == "__main__":

    pdf_path = "docs/Morning-note-and-CA-09-July--25.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The specified PDF file does not exist: {pdf_path}")
    print(f"Loading PDF from: {pdf_path}")

    pipeline = DataPipeline(pdf_path)
    pipeline.process()
    print("Data ingestion pipeline completed successfully.")



        


        