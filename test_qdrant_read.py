from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333")
col_info = client.get_collection("project_collection")
config = col_info.config.params
sparse_config = getattr(config, "sparse_vectors_config", None)
print(f"Sparse config: {sparse_config}")
print(f"Type: {type(sparse_config)}")
