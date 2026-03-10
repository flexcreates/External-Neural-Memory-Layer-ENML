from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333")
try:
    client.query_points(
        collection_name="project_collection",
        query=models.SparseVector(indices=[], values=[]),
        using="sparse"
    )
    print("Query successful!")
except Exception as e:
    print(f"Error: {e}")
