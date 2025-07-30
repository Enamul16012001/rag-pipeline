#!/usr/bin/env python3
"""
Simple RAG System Demo
======================
A simple demonstration of all RAG pipeline features.
"""

import os
from pathlib import Path
from rag_system import RAGSystem

def setup_demo_files():
    """Create demo files for testing"""
    os.makedirs('user_knowledge', exist_ok=True)
    
    # Create sample text file
    sample_content = """Enhanced RAG System Documentation
    
This system supports multiple document formats:
- PDF files with advanced text extraction
- Microsoft Word documents (DOCX and DOC)
- Plain text files (TXT) 
- CSV files with structured data

Key Features:
1. Multi-format document processing
2. Intelligent text chunking
3. Vector-based semantic search
4. File tracking and versioning
5. Comprehensive statistics

The system uses sentence transformers for embeddings and provides
natural language search across all your documents."""
    
    with open('user_knowledge/sample_doc.txt', 'w') as f:
        f.write(sample_content)
    
    # Create sample CSV
    csv_content = """content,chunk_type
"Machine learning is a subset of artificial intelligence","technical"
"Data preprocessing is crucial for model performance","best_practice"
"Regular model evaluation prevents overfitting","advice"
"Deep learning requires large datasets","fact"
"""
    
    with open('user_knowledge/ml_notes.csv', 'w') as f:
        f.write(csv_content)
    
    # Create additional test file for deletion demo
    additional_content = """This file will be used to demonstrate file deletion features.
It contains information about data science and analytics."""
    
    with open('user_knowledge/delete_test.txt', 'w') as f:
        f.write(additional_content)
    
    print("Demo files created successfully")

def main():
    """Demonstrate all RAG system features"""
    print("Enhanced RAG System Demo")
    print("=" * 30)
    
    # 1. Initialize system
    print("\n1. System Initialization")
    rag = RAGSystem()
    rag.initialize_system()
    print("System initialized successfully")
    
    # 2. Setup demo files
    print("\n2. Creating Demo Files")
    setup_demo_files()
    
    # 3. Process documents
    print("\n3. Document Processing")
    chunks_added = rag.update_user_knowledge()
    print(f"Processed documents: {chunks_added} chunks added")
    
    # 4. Show supported formats
    print("\n4. Supported Formats")
    formats = rag.get_supported_formats()
    print(f"Documents: {', '.join(formats['document_formats'])}")
    print(f"Data: {', '.join(formats['data_formats'])}")
    
    # 5. Display statistics
    print("\n5. System Statistics")
    stats = rag.get_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Files by format: {stats.get('files_by_format', {})}")
    
    # 6. List files with enhanced method
    print("\n6. File Listing (Enhanced)")
    files = rag.list_files_with_chunks("user")
    for file in files:
        print(f"{file['filename']} ({file['type']}) - {file['chunk_count']} chunks")
    
    # 7. Search demonstrations
    print("\n7. Search Functionality")
    search_queries = ["document formats", "machine learning", "data science"]
    
    for query in search_queries:
        print(f"\nSearching: '{query}'")
        results = rag.search_knowledge(query, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  Result {i} (Score: {result['similarity']:.3f})")
                print(f"  Source: {Path(result['file_path']).name}")
                preview = result['content'][:80].replace('\n', ' ').strip()
                print(f"  Preview: {preview}...")
        else:
            print("  No results found")
    
    # 8. File type specific search
    print("\n8. File Type Search")
    csv_results = rag.search_by_file_type("machine learning", "csv", top_k=2)
    print(f"CSV search results: {len(csv_results)} found")
    
    # 9. File content preview
    print("\n9. File Content Preview")
    preview = rag.get_file_content_preview("sample_doc.txt")
    if preview['success']:
        print(f"Preview length: {preview['preview_length']} characters")
        print(f"Total chunks: {preview['total_chunks']}")
    
    # 10. Enhanced file deletion
    print("\n10. Enhanced File Deletion")
    print(f"Files before deletion: {len(rag.list_files_with_chunks('user'))}")
    
    # Delete with detailed feedback
    result = rag.remove_file_completely('delete_test.txt')
    if result['success']:
        print(f"Successfully removed file")
        print(f"Chunks removed: {result['chunks_removed']}")
        print(f"Embeddings removed: {result['embeddings_removed']}")
    
    # 11. Multiple file deletion
    print("\n11. Multiple File Deletion")
    multiple_result = rag.remove_multiple_files(['ml_notes.csv'])
    print(f"Files removed: {multiple_result['successful_removals']}")
    print(f"Total chunks removed: {multiple_result['total_chunks_removed']}")
    
    # 12. Orphaned chunks cleanup
    print("\n12. Orphaned Chunks Cleanup")
    cleanup_result = rag.clean_orphaned_chunks()
    print(f"Orphaned files found: {cleanup_result['orphaned_files_found']}")
    print(f"Chunks cleaned: {cleanup_result['chunks_cleaned']}")
    
    # 13. System backup
    print("\n13. System Backup")
    backup_result = rag.backup_knowledge_base()
    if backup_result['success']:
        print(f"Backup created successfully")
        print(f"Backup location: {backup_result['backup_path']}")
    
    # 14. Final statistics
    print("\n14. Final Statistics")
    final_stats = rag.get_stats()
    print(f"Final chunks: {final_stats['total_chunks']}")
    print(f"Final embeddings: {final_stats['total_embeddings']}")
    
    print("\nDemo completed successfully!")
    print("\nQuick Start Guide:")
    print("1. Place documents in 'user_knowledge' folder")
    print("2. Run rag.update_user_knowledge() to process")
    print("3. Use rag.search_knowledge('query') to search")
    print("4. Use rag.remove_file_completely('file') to delete")

if __name__ == "__main__":
    main()