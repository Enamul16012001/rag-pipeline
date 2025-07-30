import sqlite3
import hashlib
import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from datetime import datetime
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.doc import partition_doc
from unstructured.partition.text import partition_text
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, system_folder="system_knowledge", user_folder="user_knowledge", 
                 db_path="text_chunks.db", vector_db_path="vector_chunks.db"):
        self.system_folder = system_folder
        self.user_folder = user_folder
        self.db_path = db_path
        self.vector_db_path = vector_db_path
        self.model = None
        
        # Ensure folders exist
        os.makedirs(self.system_folder, exist_ok=True)
        os.makedirs(self.user_folder, exist_ok=True)
        
    def load_embedding_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            logger.info("Loading embedding model...")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Model loaded successfully!")
        return self.model

    def create_database(self):
        """Create SQLite database and tables for storing text chunks and file tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for text chunks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                content TEXT NOT NULL,
                chunk_type TEXT,
                source TEXT DEFAULT 'pdf',
                file_path TEXT,
                file_type TEXT DEFAULT 'system',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                character_count INTEGER,
                word_count INTEGER
            )
        ''')
        
        # Create table for tracking processed files
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                file_hash TEXT,
                last_modified TIMESTAMP,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_type TEXT,
                knowledge_type TEXT DEFAULT 'system'
            )
        ''')
        
        conn.commit()
        return conn

    def create_vector_database(self):
        """Create vector database with embeddings"""
        vector_conn = sqlite3.connect(self.vector_db_path)
        vector_cursor = vector_conn.cursor()
        
        # Create table for vectors
        vector_cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                id INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                content TEXT,
                chunk_type TEXT,
                source TEXT,
                file_path TEXT,
                file_type TEXT DEFAULT 'system',
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        vector_conn.commit()
        return vector_conn

    def get_file_hash(self, file_path):
        """Generate hash of file content"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None

    def is_file_processed(self, file_path, knowledge_type='system'):
        """Check if file has been processed and if it's been modified"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT file_hash, last_modified FROM processed_files 
            WHERE file_path = ? AND knowledge_type = ?
        """, (file_path, knowledge_type))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False, "new_file"
        
        stored_hash, stored_modified = result
        current_hash = self.get_file_hash(file_path)
        
        if current_hash != stored_hash:
            return False, "modified"
        
        return True, "unchanged"

    def generate_chunk_id(self, content, source, file_path, knowledge_type):
        """Generate unique ID for each chunk based on content hash and file"""
        content_hash = hashlib.md5(f"{content}_{file_path}_{knowledge_type}".encode('utf-8')).hexdigest()[:16]
        return f"chunk_{source}_{knowledge_type}_{content_hash}"

    def remove_old_chunks(self, file_path, knowledge_type):
        """Remove chunks from old version of a file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get chunk_ids to remove from vector database
        cursor.execute("SELECT chunk_id FROM text_chunks WHERE file_path = ? AND file_type = ?", 
                      (file_path, knowledge_type))
        chunk_ids = [row[0] for row in cursor.fetchall()]
        
        # Remove from text database
        cursor.execute("DELETE FROM text_chunks WHERE file_path = ? AND file_type = ?", 
                      (file_path, knowledge_type))
        text_removed = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        # Remove from vector database
        if chunk_ids:
            vector_conn = sqlite3.connect(self.vector_db_path)
            vector_cursor = vector_conn.cursor()
            
            placeholders = ','.join(['?'] * len(chunk_ids))
            vector_cursor.execute(f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)
            vector_removed = vector_cursor.rowcount
            
            vector_conn.commit()
            vector_conn.close()
        else:
            vector_removed = 0
        
        logger.info(f"Removed {text_removed} text chunks and {vector_removed} vector embeddings for updated file: {file_path}")

    def process_pdf_file(self, file_path):
        """Process a single PDF file and return chunks"""
        logger.info(f"Processing PDF: {os.path.basename(file_path)}...")
        
        try:
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=5000,
                combine_text_under_n_chars=1000,
                new_after_n_chars=2000,
                include_page_breaks=True,
                languages=["eng"],
            )
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []

    def process_docx_file(self, file_path):
        """Process a single DOCX file and return chunks"""
        logger.info(f"Processing DOCX: {os.path.basename(file_path)}...")
        
        try:
            chunks = partition_docx(
                filename=file_path,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=5000,
                combine_text_under_n_chars=1000,
                new_after_n_chars=2000,
                languages=["eng"],
            )
            return chunks
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            return []

    def process_doc_file(self, file_path):
        """Process a single DOC file and return chunks"""
        logger.info(f"Processing DOC: {os.path.basename(file_path)}...")
        
        try:
            chunks = partition_doc(
                filename=file_path,
                chunking_strategy="by_title",
                max_characters=5000,
                combine_text_under_n_chars=1000,
                new_after_n_chars=2000,
                languages=["eng"],
            )
            return chunks
        except Exception as e:
            logger.error(f"Error processing DOC {file_path}: {e}")
            return []

    def process_txt_file(self, file_path):
        """Process a single TXT file and return chunks"""
        logger.info(f"Processing TXT: {os.path.basename(file_path)}...")
        
        try:
            chunks = partition_text(
                filename=file_path,
                chunking_strategy="by_title",
                max_characters=5000,
                combine_text_under_n_chars=1000,
                new_after_n_chars=2000,
                languages=["eng"],
            )
            return chunks
        except Exception as e:
            logger.error(f"Error processing TXT {file_path}: {e}")
            return []

    def process_document_file(self, file_path):
        """Process any supported document file based on its extension"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.process_pdf_file(file_path), 'pdf'
        elif file_ext == '.docx':
            return self.process_docx_file(file_path), 'docx'
        elif file_ext == '.doc':
            return self.process_doc_file(file_path), 'doc'
        elif file_ext == '.txt':
            return self.process_txt_file(file_path), 'txt'
        else:
            logger.warning(f"Unsupported file format: {file_ext}")
            return [], 'unknown'

    def save_chunks_to_db(self, chunks, file_path, knowledge_type, source_type):
        """Save text chunks to SQLite database"""
        conn = self.create_database()
        cursor = conn.cursor()
        
        # Separate tables from texts
        tables = []
        texts = []
        
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            if "CompositeElement" in str(type(chunk)) or "Text" in str(type(chunk)) or "NarrativeText" in str(type(chunk)):
                texts.append(chunk)
        
        saved_count = 0
        
        # Save text chunks to database
        for chunk in texts:
            try:
                content = str(chunk).strip()
                
                if not content or len(content.strip()) < 10:
                    continue
                
                chunk_id = self.generate_chunk_id(content, source_type, file_path, knowledge_type)
                chunk_type = str(type(chunk).__name__)
                character_count = len(content)
                word_count = len(content.split())
                
                cursor.execute('''
                    INSERT OR IGNORE INTO text_chunks 
                    (chunk_id, content, chunk_type, source, file_path, file_type, character_count, word_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (chunk_id, content, chunk_type, source_type, file_path, knowledge_type, character_count, word_count))
                
                if cursor.rowcount > 0:
                    saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        return saved_count

    def save_csv_chunks_to_db(self, file_path, knowledge_type):
        """Process CSV file and save chunks"""
        try:
            df = pd.read_csv(file_path)
            
            required_columns = ['content']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"CSV must contain 'content' column. Available columns: {list(df.columns)}")
                return 0
            
            conn = self.create_database()
            cursor = conn.cursor()
            
            saved_count = 0
            
            for _, row in df.iterrows():
                try:
                    content = str(row['content']).strip()
                    
                    if not content or len(content.strip()) < 10:
                        continue
                    
                    chunk_id = self.generate_chunk_id(content, "csv", file_path, knowledge_type)
                    chunk_type = str(row.get('chunk_type', 'custom'))
                    character_count = len(content)
                    word_count = len(content.split())
                    
                    cursor.execute('''
                        INSERT OR IGNORE INTO text_chunks 
                        (chunk_id, content, chunk_type, source, file_path, file_type, character_count, word_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (chunk_id, content, chunk_type, "csv", file_path, knowledge_type, character_count, word_count))
                    
                    if cursor.rowcount > 0:
                        saved_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing CSV row: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            return saved_count
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            return 0

    def update_file_tracking(self, file_path, file_type, knowledge_type):
        """Update file tracking information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        file_hash = self.get_file_hash(file_path)
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        cursor.execute('''
            INSERT OR REPLACE INTO processed_files 
            (file_path, file_hash, last_modified, file_type, knowledge_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (file_path, file_hash, last_modified, file_type, knowledge_type))
        
        conn.commit()
        conn.close()

    def create_embeddings_for_new_chunks(self, knowledge_type=None):
        """Create embeddings for new chunks"""
        # Ensure vector database exists first
        vector_conn = self.create_vector_database()
        vector_conn.close()
        
        # Get chunks that don't have embeddings yet
        text_conn = sqlite3.connect(self.db_path)
        text_cursor = text_conn.cursor()
        
        # Check if vector database table exists
        vector_conn = sqlite3.connect(self.vector_db_path)
        vector_cursor = vector_conn.cursor()
        
        # Check if chunk_embeddings table exists, if not create it
        vector_cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='chunk_embeddings'
        """)
        
        if not vector_cursor.fetchone():
            vector_conn.close()
            vector_conn = self.create_vector_database()
            vector_cursor = vector_conn.cursor()
        
        # Get existing embedding chunk_ids
        vector_cursor.execute("SELECT chunk_id FROM chunk_embeddings")
        existing_chunk_ids = {row[0] for row in vector_cursor.fetchall()}
        vector_conn.close()
        
        # Get all chunks
        if knowledge_type:
            text_cursor.execute("""
                SELECT id, chunk_id, content, chunk_type, source, file_path, file_type
                FROM text_chunks WHERE file_type = ?
            """, (knowledge_type,))
        else:
            text_cursor.execute("""
                SELECT id, chunk_id, content, chunk_type, source, file_path, file_type
                FROM text_chunks
            """)
        
        all_chunks = text_cursor.fetchall()
        text_conn.close()
        
        # Filter chunks that don't have embeddings
        chunks = [chunk for chunk in all_chunks if chunk[1] not in existing_chunk_ids]
        
        if not chunks:
            logger.info("No new chunks to embed")
            return 0
        
        logger.info(f"Creating embeddings for {len(chunks)} new chunks...")
        
        # Load model
        model = self.load_embedding_model()
        
        # Create vector database connection
        vector_conn = sqlite3.connect(self.vector_db_path)
        vector_cursor = vector_conn.cursor()
        
        # Process in batches
        batch_size = 32
        total_processed = 0
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
            batch = chunks[i:i + batch_size]
            batch_content = [chunk[2] for chunk in batch]  # content is at index 2
            
            embeddings = model.encode(batch_content, show_progress_bar=False)
            
            for j, (chunk_id, chunk_uuid, content, chunk_type, source, file_path_db, file_type) in enumerate(batch):
                embedding_blob = pickle.dumps(embeddings[j])
                
                vector_cursor.execute('''
                    INSERT OR REPLACE INTO chunk_embeddings 
                    (id, chunk_id, content, chunk_type, source, file_path, file_type, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (chunk_id, chunk_uuid, content, chunk_type, source, file_path_db, file_type, embedding_blob))
            
            total_processed += len(batch)
            
            if i % (batch_size * 10) == 0:
                vector_conn.commit()
        
        vector_conn.commit()
        vector_conn.close()
        
        logger.info(f"Created embeddings for {total_processed} new chunks")
        return total_processed

    def process_files_in_folder(self, folder_path, knowledge_type):
        """Process all supported files in a folder"""
        # Get all supported file types
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        docx_files = glob.glob(os.path.join(folder_path, "*.docx"))
        doc_files = glob.glob(os.path.join(folder_path, "*.doc"))
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        # Combine all document files (excluding CSV for now)
        document_files = pdf_files + docx_files + doc_files + txt_files
        
        total_saved = 0
        processed_files = 0
        
        # Process document files (PDF, DOCX, DOC, TXT)
        for doc_file in document_files:
            is_processed, status = self.is_file_processed(doc_file, knowledge_type)
            
            if not is_processed:
                if status == "modified":
                    logger.info(f"File modified: {os.path.basename(doc_file)}")
                    self.remove_old_chunks(doc_file, knowledge_type)
                
                try:
                    chunks, source_type = self.process_document_file(doc_file)
                    if chunks:
                        saved_count = self.save_chunks_to_db(chunks, doc_file, knowledge_type, source_type)
                        self.update_file_tracking(doc_file, source_type, knowledge_type)
                        total_saved += saved_count
                        processed_files += 1
                        logger.info(f"Processed {os.path.basename(doc_file)}: {saved_count} chunks saved")
                except Exception as e:
                    logger.error(f"Error processing {doc_file}: {e}")
        
        # Process CSV files
        for csv_file in csv_files:
            is_processed, status = self.is_file_processed(csv_file, knowledge_type)
            
            if not is_processed:
                if status == "modified":
                    logger.info(f"CSV file modified: {os.path.basename(csv_file)}")
                    self.remove_old_chunks(csv_file, knowledge_type)
                
                try:
                    saved_count = self.save_csv_chunks_to_db(csv_file, knowledge_type)
                    if saved_count > 0:
                        self.update_file_tracking(csv_file, "csv", knowledge_type)
                        total_saved += saved_count
                        processed_files += 1
                        logger.info(f"Processed {os.path.basename(csv_file)}: {saved_count} chunks saved")
                except Exception as e:
                    logger.error(f"Error processing {csv_file}: {e}")
        
        # Create embeddings for new chunks
        if total_saved > 0:
            self.create_embeddings_for_new_chunks(knowledge_type)
        
        logger.info(f"Processed {processed_files} files with {total_saved} chunks in {knowledge_type} knowledge")
        return total_saved

    def initialize_system(self):
        """Initialize the RAG system for the first time"""
        logger.info("Initializing RAG system...")
        
        # Create databases
        self.create_database()
        self.create_vector_database()
        
        # Process system knowledge
        system_chunks = self.process_files_in_folder(self.system_folder, "system")
        
        # Process user knowledge
        user_chunks = self.process_files_in_folder(self.user_folder, "user")
        
        logger.info(f"RAG system initialized with {system_chunks} system chunks and {user_chunks} user chunks")

    def update_system_knowledge(self):
        """Update system knowledge (hidden from user)"""
        return self.process_files_in_folder(self.system_folder, "system")

    def update_user_knowledge(self):
        """Update user knowledge (visible to user)"""
        return self.process_files_in_folder(self.user_folder, "user")

    def remove_user_file(self, filename):
        """Remove a specific user file from the system (original method)"""
        file_path = os.path.join(self.user_folder, filename)
        
        if os.path.exists(file_path):
            # Remove chunks from databases
            self.remove_old_chunks(file_path, "user")
            
            # Remove from processed files tracking
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM processed_files WHERE file_path = ? AND knowledge_type = ?", 
                          (file_path, "user"))
            conn.commit()
            conn.close()
            
            logger.info(f"Removed user file: {filename}")

    def remove_file_completely(self, filename, knowledge_type="user"):
        """
        Remove a specific file completely from the system with detailed feedback
        
        Args:
            filename (str): Name of the file to remove
            knowledge_type (str): "user" or "system"
        
        Returns:
            dict: Detailed information about what was removed
        """
        if knowledge_type == "user":
            file_path = os.path.join(self.user_folder, filename)
        else:
            file_path = os.path.join(self.system_folder, filename)
        
        # Check if file exists in tracking
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM processed_files 
            WHERE file_path = ? AND knowledge_type = ?
        """, (file_path, knowledge_type))
        
        file_tracked = cursor.fetchone()[0] > 0
        
        if not file_tracked:
            conn.close()
            return {
                "success": False,
                "message": f"File '{filename}' not found in knowledge base",
                "chunks_removed": 0,
                "embeddings_removed": 0
            }
        
        # Get chunks count before removal
        cursor.execute("""
            SELECT COUNT(*) FROM text_chunks 
            WHERE file_path = ? AND file_type = ?
        """, (file_path, knowledge_type))
        
        chunks_count = cursor.fetchone()[0]
        conn.close()
        
        # Get embeddings count before removal
        vector_conn = sqlite3.connect(self.vector_db_path)
        vector_cursor = vector_conn.cursor()
        
        vector_cursor.execute("""
            SELECT COUNT(*) FROM chunk_embeddings 
            WHERE file_path = ? AND file_type = ?
        """, (file_path, knowledge_type))
        
        embeddings_count = vector_cursor.fetchone()[0]
        vector_conn.close()
        
        # Remove chunks using existing method
        self.remove_old_chunks(file_path, knowledge_type)
        
        # Remove from processed files tracking
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM processed_files 
            WHERE file_path = ? AND knowledge_type = ?
        """, (file_path, knowledge_type))
        conn.commit()
        conn.close()
        
        # Remove physical file if it exists
        physical_file_removed = False
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                physical_file_removed = True
            except Exception as e:
                logger.error(f"Could not remove physical file {file_path}: {e}")
        
        logger.info(f"Completely removed file: {filename}")
        
        return {
            "success": True,
            "message": f"Successfully removed '{filename}' from knowledge base",
            "chunks_removed": chunks_count,
            "embeddings_removed": embeddings_count,
            "physical_file_removed": physical_file_removed,
            "file_path": file_path
        }

    def list_files_with_chunks(self, knowledge_type=None):
        """
        List all files in the knowledge base with their chunk counts
        
        Args:
            knowledge_type (str, optional): "user", "system", or None for all
        
        Returns:
            list: List of files with detailed information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if knowledge_type:
            cursor.execute("""
                SELECT pf.file_path, pf.file_type, pf.knowledge_type, 
                       pf.processed_at, COUNT(tc.id) as chunk_count
                FROM processed_files pf
                LEFT JOIN text_chunks tc ON pf.file_path = tc.file_path
                WHERE pf.knowledge_type = ?
                GROUP BY pf.file_path, pf.file_type, pf.knowledge_type, pf.processed_at
                ORDER BY pf.processed_at DESC
            """, (knowledge_type,))
        else:
            cursor.execute("""
                SELECT pf.file_path, pf.file_type, pf.knowledge_type, 
                       pf.processed_at, COUNT(tc.id) as chunk_count
                FROM processed_files pf
                LEFT JOIN text_chunks tc ON pf.file_path = tc.file_path
                GROUP BY pf.file_path, pf.file_type, pf.knowledge_type, pf.processed_at
                ORDER BY pf.knowledge_type, pf.processed_at DESC
            """)
        
        files_info = []
        for row in cursor.fetchall():
            file_path, file_type, knowledge_type, processed_at, chunk_count = row
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            files_info.append({
                "filename": filename,
                "file_path": file_path,
                "type": file_type.upper(),
                "knowledge_type": knowledge_type,
                "size": file_size,
                "processed_at": processed_at,
                "chunk_count": chunk_count,
                "exists": os.path.exists(file_path)
            })
        
        conn.close()
        return files_info

    def remove_multiple_files(self, filenames, knowledge_type="user"):
        """
        Remove multiple files at once
        
        Args:
            filenames (list): List of filenames to remove
            knowledge_type (str): "user" or "system"
        
        Returns:
            dict: Summary of removal operation
        """
        results = []
        total_chunks = 0
        total_embeddings = 0
        successful_removals = 0
        
        for filename in filenames:
            result = self.remove_file_completely(filename, knowledge_type)
            results.append(result)
            
            if result["success"]:
                successful_removals += 1
                total_chunks += result["chunks_removed"]
                total_embeddings += result["embeddings_removed"]
        
        return {
            "total_files_requested": len(filenames),
            "successful_removals": successful_removals,
            "failed_removals": len(filenames) - successful_removals,
            "total_chunks_removed": total_chunks,
            "total_embeddings_removed": total_embeddings,
            "detailed_results": results
        }

    def clean_orphaned_chunks(self):
        """
        Remove chunks that belong to files no longer tracked in processed_files
        This is useful for cleaning up after manual file deletions
        
        Returns:
            dict: Information about cleanup operation
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find chunks without corresponding tracked files
        cursor.execute("""
            SELECT tc.file_path, COUNT(*) as chunk_count
            FROM text_chunks tc
            LEFT JOIN processed_files pf ON tc.file_path = pf.file_path
            WHERE pf.file_path IS NULL
            GROUP BY tc.file_path
        """)
        
        orphaned_files = cursor.fetchall()
        
        if not orphaned_files:
            conn.close()
            return {
                "orphaned_files_found": 0,
                "chunks_cleaned": 0,
                "embeddings_cleaned": 0,
                "cleaned_files": []
            }
        
        total_chunks_cleaned = 0
        total_embeddings_cleaned = 0
        
        # Remove orphaned chunks
        for file_path, chunk_count in orphaned_files:
            # Get chunk_ids for this file before removing from text_chunks
            cursor.execute("SELECT chunk_id FROM text_chunks WHERE file_path = ?", (file_path,))
            chunk_ids = [row[0] for row in cursor.fetchall()]
            
            # Remove from text chunks
            cursor.execute("DELETE FROM text_chunks WHERE file_path = ?", (file_path,))
            chunks_removed = cursor.rowcount
            total_chunks_cleaned += chunks_removed
            
            # Remove from embeddings
            if chunk_ids:
                vector_conn = sqlite3.connect(self.vector_db_path)
                vector_cursor = vector_conn.cursor()
                
                placeholders = ','.join(['?'] * len(chunk_ids))
                vector_cursor.execute(f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)
                embeddings_removed = vector_cursor.rowcount
                total_embeddings_cleaned += embeddings_removed
                
                vector_conn.commit()
                vector_conn.close()
            
            logger.info(f"Cleaned orphaned chunks for: {file_path}")
        
        conn.commit()
        conn.close()
        
        return {
            "orphaned_files_found": len(orphaned_files),
            "chunks_cleaned": total_chunks_cleaned,
            "embeddings_cleaned": total_embeddings_cleaned,
            "cleaned_files": [fp for fp, _ in orphaned_files]
        }

    def search_knowledge(self, query, top_k=5, knowledge_type=None):
        """Search for similar chunks using cosine similarity"""
        try:
            model = self.load_embedding_model()
            query_embedding = model.encode([query])[0]
            
            # Ensure vector database exists
            if not os.path.exists(self.vector_db_path):
                logger.warning("Vector database not found, creating...")
                self.create_vector_database()
                return []
            
            conn = sqlite3.connect(self.vector_db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='chunk_embeddings'
            """)
            
            if not cursor.fetchone():
                logger.warning("chunk_embeddings table not found")
                conn.close()
                return []
            
            if knowledge_type:
                cursor.execute("""
                    SELECT chunk_id, content, source, chunk_type, file_path, file_type, embedding 
                    FROM chunk_embeddings WHERE file_type = ?
                """, (knowledge_type,))
            else:
                cursor.execute("""
                    SELECT chunk_id, content, source, chunk_type, file_path, file_type, embedding 
                    FROM chunk_embeddings
                """)
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                logger.info("No embeddings found in database")
                return []
            
            similarities = []
            for chunk_id, content, source, chunk_type, file_path, file_type, embedding_blob in results:
                try:
                    embedding = pickle.loads(embedding_blob)
                    
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    
                    similarities.append({
                        'chunk_id': chunk_id,
                        'content': content,
                        'source': source,
                        'chunk_type': chunk_type,
                        'file_path': file_path,
                        'file_type': file_type,
                        'similarity': similarity
                    })
                except Exception as e:
                    logger.error(f"Error processing embedding for chunk {chunk_id}: {e}")
                    continue
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in knowledge search: {e}")
            return []

    def get_stats(self):
        """Get comprehensive statistics about the knowledge base"""
        try:
            # Ensure databases exist
            if not os.path.exists(self.db_path):
                return {"error": "Text database not found"}
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if text_chunks table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='text_chunks'
            """)
            
            if not cursor.fetchone():
                conn.close()
                return {"error": "text_chunks table not found"}
            
            # Total chunks by type
            cursor.execute("SELECT file_type, COUNT(*) FROM text_chunks GROUP BY file_type")
            chunks_by_type = dict(cursor.fetchall())
            
            # Total chunks by source
            cursor.execute("SELECT source, COUNT(*) FROM text_chunks GROUP BY source")
            chunks_by_source = dict(cursor.fetchall())
            
            # Total processed files by type
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='processed_files'
            """)
            
            if cursor.fetchone():
                cursor.execute("SELECT knowledge_type, COUNT(*) FROM processed_files GROUP BY knowledge_type")
                files_by_type = dict(cursor.fetchall())
                
                # Get file types breakdown
                cursor.execute("SELECT file_type, COUNT(*) FROM processed_files GROUP BY file_type")
                files_by_format = dict(cursor.fetchall())
            else:
                files_by_type = {}
                files_by_format = {}
            
            conn.close()
            
            # Vector database stats
            embeddings_by_type = {}
            total_embeddings = 0
            
            if os.path.exists(self.vector_db_path):
                vector_conn = sqlite3.connect(self.vector_db_path)
                vector_cursor = vector_conn.cursor()
                
                # Check if chunk_embeddings table exists
                vector_cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='chunk_embeddings'
                """)
                
                if vector_cursor.fetchone():
                    vector_cursor.execute("SELECT COUNT(*) FROM chunk_embeddings")
                    total_embeddings = vector_cursor.fetchone()[0]
                    
                    vector_cursor.execute("SELECT file_type, COUNT(*) FROM chunk_embeddings GROUP BY file_type")
                    embeddings_by_type = dict(vector_cursor.fetchall())
                
                vector_conn.close()
            
            return {
                "total_chunks": sum(chunks_by_type.values()),
                "total_embeddings": total_embeddings,
                "chunks_by_type": chunks_by_type,
                "chunks_by_source": chunks_by_source,
                "files_by_type": files_by_type,
                "files_by_format": files_by_format,
                "embeddings_by_type": embeddings_by_type,
                "system_chunks": chunks_by_type.get("system", 0),
                "user_chunks": chunks_by_type.get("user", 0),
                "supported_formats": ["PDF", "DOCX", "DOC", "TXT", "CSV"]
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    def get_user_files_info(self):
        """Get information about user uploaded files only"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path, file_type, processed_at, 
                       COUNT(tc.id) as chunk_count
                FROM processed_files pf
                LEFT JOIN text_chunks tc ON pf.file_path = tc.file_path
                WHERE pf.knowledge_type = 'user'
                GROUP BY pf.file_path, pf.file_type, pf.processed_at
                ORDER BY pf.processed_at DESC
            """)
            
            files_info = []
            for row in cursor.fetchall():
                file_path, file_type, processed_at, chunk_count = row
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                
                files_info.append({
                    "filename": filename,
                    "type": file_type.upper(),
                    "size": file_size,
                    "modified": processed_at,
                    "chunk_count": chunk_count
                })
            
            conn.close()
            return files_info
            
        except Exception as e:
            logger.error(f"Error getting user files info: {e}")
            return []

    def get_supported_formats(self):
        """Get list of supported file formats"""
        return {
            "document_formats": [".pdf", ".docx", ".doc", ".txt"],
            "data_formats": [".csv"],
            "total_supported": 5
        }

    def search_by_file_type(self, query, file_type, top_k=5):
        """
        Search within specific file types only
        
        Args:
            query (str): Search query
            file_type (str): File type to search in (pdf, docx, doc, txt, csv)
            top_k (int): Number of results to return
        
        Returns:
            list: Search results from specified file type only
        """
        try:
            model = self.load_embedding_model()
            query_embedding = model.encode([query])[0]
            
            if not os.path.exists(self.vector_db_path):
                return []
            
            conn = sqlite3.connect(self.vector_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT chunk_id, content, source, chunk_type, file_path, file_type, embedding 
                FROM chunk_embeddings WHERE source = ?
            """, (file_type,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return []
            
            similarities = []
            for chunk_id, content, source, chunk_type, file_path, file_type, embedding_blob in results:
                try:
                    embedding = pickle.loads(embedding_blob)
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    
                    similarities.append({
                        'chunk_id': chunk_id,
                        'content': content,
                        'source': source,
                        'chunk_type': chunk_type,
                        'file_path': file_path,
                        'file_type': file_type,
                        'similarity': similarity
                    })
                except Exception as e:
                    logger.error(f"Error processing embedding: {e}")
                    continue
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in file type search: {e}")
            return []

    def get_file_content_preview(self, filename, knowledge_type="user", max_chars=500):
        """
        Get a preview of file content from chunks
        
        Args:
            filename (str): Name of the file
            knowledge_type (str): "user" or "system"
            max_chars (int): Maximum characters to return
        
        Returns:
            dict: File preview information
        """
        if knowledge_type == "user":
            file_path = os.path.join(self.user_folder, filename)
        else:
            file_path = os.path.join(self.system_folder, filename)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT content, chunk_type, character_count, word_count
                FROM text_chunks 
                WHERE file_path = ? AND file_type = ?
                ORDER BY id
                LIMIT 5
            """, (file_path, knowledge_type))
            
            chunks = cursor.fetchall()
            conn.close()
            
            if not chunks:
                return {
                    "success": False,
                    "message": f"No content found for file '{filename}'"
                }
            
            # Combine first few chunks for preview
            preview_content = ""
            total_chars = 0
            total_words = 0
            
            for content, chunk_type, char_count, word_count in chunks:
                total_chars += char_count
                total_words += word_count
                
                if len(preview_content) + len(content) <= max_chars:
                    preview_content += content + "\n\n"
                else:
                    remaining_chars = max_chars - len(preview_content)
                    if remaining_chars > 0:
                        preview_content += content[:remaining_chars] + "..."
                    break
            
            return {
                "success": True,
                "filename": filename,
                "preview": preview_content.strip(),
                "total_chunks": len(chunks),
                "total_characters": total_chars,
                "total_words": total_words,
                "preview_length": len(preview_content)
            }
            
        except Exception as e:
            logger.error(f"Error getting file preview: {e}")
            return {
                "success": False,
                "message": f"Error retrieving preview: {str(e)}"
            }

    def backup_knowledge_base(self, backup_path="backup"):
        """
        Create a backup of the entire knowledge base
        
        Args:
            backup_path (str): Directory to store backup files
        
        Returns:
            dict: Backup operation results
        """
        import shutil
        from datetime import datetime
        
        try:
            # Create backup directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(backup_path, f"rag_backup_{timestamp}")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup databases
            if os.path.exists(self.db_path):
                shutil.copy2(self.db_path, os.path.join(backup_dir, "text_chunks.db"))
            
            if os.path.exists(self.vector_db_path):
                shutil.copy2(self.vector_db_path, os.path.join(backup_dir, "vector_chunks.db"))
            
            # Backup knowledge folders
            if os.path.exists(self.system_folder):
                shutil.copytree(self.system_folder, os.path.join(backup_dir, "system_knowledge"))
            
            if os.path.exists(self.user_folder):
                shutil.copytree(self.user_folder, os.path.join(backup_dir, "user_knowledge"))
            
            # Get stats for backup report
            stats = self.get_stats()
            
            # Create backup info file
            backup_info = {
                "backup_timestamp": timestamp,
                "total_chunks": stats.get("total_chunks", 0),
                "total_embeddings": stats.get("total_embeddings", 0),
                "system_chunks": stats.get("system_chunks", 0),
                "user_chunks": stats.get("user_chunks", 0),
                "files_by_format": stats.get("files_by_format", {}),
                "backup_path": backup_dir
            }
            
            import json
            with open(os.path.join(backup_dir, "backup_info.json"), 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            logger.info(f"Knowledge base backed up to: {backup_dir}")
            
            return {
                "success": True,
                "backup_path": backup_dir,
                "timestamp": timestamp,
                "total_chunks": stats.get("total_chunks", 0),
                "total_embeddings": stats.get("total_embeddings", 0)
            }
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }