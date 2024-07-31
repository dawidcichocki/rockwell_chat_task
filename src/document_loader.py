import re
from typing import List
from datetime import datetime
import fitz
import logging
from dateutil.parser import parse
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

ALLOWED_EXTENSIONS = {'.pdf'}


class PDFDocumentLoader:
    def __init__(self, file_paths: List[str]):
        self._file_paths = file_paths
        self._documents = []
        self.splits = []

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a pdf document from a file path
        """
        if file_path.endswith('.pdf'):
            documents = PyMuPDFLoader(file_path).load()
            logging.info(f"Loaded {len(documents)} documents from {file_path}")
            self._documents.extend(documents)
            return documents
        else:
            raise ValueError("File extension not allowed. Only .pdf files are allowed.")

    def split_document_into_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Splits a document into chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return text_splitter.split_documents(documents)

    def create_splits_with_custom_metadata(self, splits: List[Document]) -> List[Document]:
        """
        Creates sections from the splits
        """
        file_path = splits[0].metadata.get('file_path')
        file_name = file_path.split('/')[-1]
        file_id = re.sub("[^0-9a-zA-Z-_]", "_", file_name)
        document = fitz.open(file_path)

        sections = []

        for i, split in enumerate(splits):
            if not split.page_content:
                logging.info(f"Empty page content for split {i} in document {file_name}")
                continue
            page_number = split.metadata.get('page')
            page_content = document.load_page(page_number)
            text_instances = page_content.search_for(split.page_content)

            if text_instances:
                bbox = list(text_instances[0])
                for (x0, y0, x1, y1) in text_instances[1:]:
                    bbox[0] = min(bbox[0], x0)
                    bbox[1] = min(bbox[1], y0)
                    bbox[2] = max(bbox[2], x1)
                    bbox[3] = max(bbox[3], y1)

                bbox = tuple([int(x) for x in bbox])
            else:
                bbox = (0, 0, 0, 0)
                logging.error(f"Could not find text instances for page {page_number} in document {file_name}")

            try:
                modification_date = document.metadata.get('modDate')[2:-7]
                modification_date = parse(modification_date).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            except Exception:
                modification_date = datetime.now().isoformat()

            section = Document(
                page_content=split.page_content,
                metadata={
                    "id": f"{file_id}-{i}",
                    "file": file_name,
                    "page": page_number,
                    "bbox": bbox,
                    "file_path": file_path,
                    "last_modified": modification_date,
                }
            )
            sections.append(section)

        logging.info(f"Created {len(sections)} chunks from {file_name}")
        return sections

    def load_and_split_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a pdf document from a file path and splits it into chunks
        """
        documents = self.load_pdf(file_path)
        chunks = self.split_document_into_chunks(documents)
        splits = self.create_splits_with_custom_metadata(chunks)

        return splits

    def load_documents(self) -> List[Document]:
        """
        Loads the documents from the file paths
        """
        for file_path in self._file_paths:
            try:
                self.splits.extend(self.load_and_split_pdf(file_path))
            except Exception as e:
                logging.error(f"Error loading document {file_path.split('/')[-1]}: {e}")
        return self.splits