from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333")
try:
    client.update_collection(
        collection_name="project_collection",
        sparse_vectors_config={"sparse": models.SparseVectorParams()}
    )
    print("Update successful")
except Exception as e:
    print(f"Error: {e}")
