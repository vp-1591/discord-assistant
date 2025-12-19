# Assuming 'create_store_rag.vector_store' is your Chroma instance
# and 'document_ids' contains the IDs you just added.
import create_store_rag

target_id = '20fc119e-dd70-4b73-983a-17af59c0477a'  # Let's look at the first one

# Retrieve the document by ID
result = create_store_rag.vector_store.get(ids=[target_id])

# The result is a dictionary with keys like 'ids', 'documents', 'metadatas'
print(f"--- Document ID: {target_id} ---")
print(f"Content:\n{result['documents'][0]}")
print(f"Metadata:\n{result['metadatas'][0]}")