# rag-pipeline

A comprehensive **Retrieval-Augmented Generation (RAG)** system supporting multiple document formats with semantic search capabilities.

## Features

### Document Support
- **PDF** files with advanced text extraction
- **Microsoft Word** documents (DOCX and DOC)
- **Plain text** files (TXT)
- **CSV** files with structured data

### Core Capabilities
- Semantic chunking with configurable strategies
- Vector embeddings using sentence transformers
- Similarity search with cosine similarity
- File versioning and change detection
- Comprehensive file management
- Database maintenance tools

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/enhanced-rag-system.git
cd enhanced-rag-system
pip install -r requirements.txt
```

### Run Demo
```bash
python main.py
```

### Basic Usage
```python
from rag_system import RAGSystem

# Initialize
rag = RAGSystem()
rag.initialize_system()

# Add documents (place files in user_knowledge folder)
chunks_added = rag.update_user_knowledge()

# Search
results = rag.search_knowledge("your query", top_k=5)
for result in results:
    print(f"Score: {result['similarity']:.3f}")
    print(f"Content: {result['content'][:200]}...")
```

## Key Methods

### Document Processing
```python
# Process new/updated files
rag.update_user_knowledge()
rag.update_system_knowledge()

# Get supported formats
formats = rag.get_supported_formats()
```

### Search & Retrieval
```python
# General search
results = rag.search_knowledge("query", top_k=5)

# Search by file type
results = rag.search_by_file_type("query", "pdf", top_k=3)

# Search by knowledge type
results = rag.search_knowledge("query", knowledge_type="user")
```

### File Management
```python
# List files with details
files = rag.list_files_with_chunks("user")

# Remove single file with feedback
result = rag.remove_file_completely("document.pdf")
print(f"Removed {result['chunks_removed']} chunks")

# Remove multiple files
rag.remove_multiple_files(["doc1.pdf", "doc2.txt"])

# Get file content preview
preview = rag.get_file_content_preview("document.txt")
```

### Maintenance
```python
# Clean orphaned data
cleanup = rag.clean_orphaned_chunks()
print(f"Cleaned {cleanup['chunks_cleaned']} orphaned chunks")

# Backup system
backup = rag.backup_knowledge_base()

# Get comprehensive statistics
stats = rag.get_stats()
```

## Project Structure

```
enhanced-rag-system/
├── rag_system.py          # Core RAG implementation
├── main.py               # Demo application
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── system_knowledge/    # System documents
├── user_knowledge/      # User documents
├── text_chunks.db      # Text database
├── vector_chunks.db    # Embeddings database
└── rag_system.log     # Application logs
```

## Configuration

### RAGSystem Parameters
```python
rag = RAGSystem(
    system_folder="system_knowledge",
    user_folder="user_knowledge",
    db_path="text_chunks.db",
    vector_db_path="vector_chunks.db"
)
```

### Chunking Settings
- Max characters per chunk: 5,000
- Combine text under: 1,000 characters
- New chunk after: 2,000 characters
- Strategy: By title/section

## File Formats

### CSV Format
Include a `content` column with your text data:
```csv
content,chunk_type
"Your document content here","article"
"Another piece of content","note"
```

### Supported Extensions
- Documents: `.pdf`, `.docx`, `.doc`, `.txt`
- Data: `.csv`

## Performance

- **Processing**: 1-10 documents/second
- **Memory**: 2-4GB RAM typical usage
- **Storage**: 10-50MB per 1000 chunks
- **First run**: Model download (~500MB)

## Common Use Cases

### Document Search
```python
# Search across all documents
results = rag.search_knowledge("project timeline")

# Search only in PDFs
pdf_results = rag.search_by_file_type("budget", "pdf")
```

### File Management
```python
# Add new documents
# 1. Place files in user_knowledge/
# 2. Process them
rag.update_user_knowledge()

# Remove outdated documents
rag.remove_file_completely("old_report.pdf")

# Clean up after manual deletions
rag.clean_orphaned_chunks()
```

### System Maintenance
```python
# Weekly maintenance
cleanup = rag.clean_orphaned_chunks()
backup = rag.backup_knowledge_base()
stats = rag.get_stats()
```

## Error Handling

The system provides detailed error information:
```python
result = rag.remove_file_completely("nonexistent.pdf")
if not result['success']:
    print(result['message'])  # Clear error description
```

## Troubleshooting

### Common Issues
1. **No search results**: Check if documents are processed with `rag.get_stats()`
2. **Processing errors**: Verify file permissions and formats
3. **Memory issues**: Process files in smaller batches

### Logs
Check `rag_system.log` for detailed information:
```bash
tail -f rag_system.log
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push branch: `git push origin feature/new-feature`
5. Create Pull Request
