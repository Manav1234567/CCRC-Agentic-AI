import pandas as pd
import numpy as np
import os
from pymilvus import MilvusClient, DataType

class DatabaseManager:
    def __init__(self):
        # Use the internal Docker URL by default
        self.uri = os.getenv("MILVUS_URI", "http://milvus:19530")
        self.collection_name = "climate_articles"
        self.data_dir = "/app/data"  # Path inside Docker

    def get_client(self):
        return MilvusClient(uri=self.uri)

    def get_count(self):
        """Returns the number of articles in the database."""
        try:
            client = self.get_client()
            if not client.has_collection(self.collection_name):
                return 0
            res = client.query(collection_name=self.collection_name, filter="", output_fields=["count(*)"])
            return res[0]["count(*)"]
        except Exception as e:
            print(f"DB Connection Check Failed: {e}")
            return -1

    def ingest_default_data(self):
        """Reads .parquet and .npy files from /app/data and inserts them."""
        parquet_path = os.path.join(self.data_dir, "climate_news_data.parquet")
        npy_path = os.path.join(self.data_dir, "climate_vectors.npy")

        if not os.path.exists(parquet_path) or not os.path.exists(npy_path):
            return False, "Data files not found in /app/data"

        client = self.get_client()
        
        # 1. Recreate Collection
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)
        
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2560)
        # Add index immediately
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        
        client.create_collection(
            collection_name=self.collection_name, 
            schema=schema, 
            index_params=index_params
        )

        # 2. Load and Insert
        df = pd.read_parquet(parquet_path)
        vectors = np.load(npy_path)
        
        data_to_insert = []
        for idx, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else ""
            entry = {
                "vector": vectors[idx].tolist(),
                "text": str(row['body']),
                "title": str(row['title']),
                "category": str(row['category']),
                "date": date_str,
                "tags": str(row['tags'])
            }
            data_to_insert.append(entry)

        # Batch Insert
        batch_size = 100
        for i in range(0, len(data_to_insert), batch_size):
            client.insert(collection_name=self.collection_name, data=data_to_insert[i:i+batch_size])
            
        return True, f"Successfully inserted {len(data_to_insert)} records."