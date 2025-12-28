import pandas as pd
import numpy as np
import os
from pymilvus import MilvusClient, DataType

class DatabaseManager:
    def __init__(self):
        self.uri = os.getenv("MILVUS_URI", "http://milvus:19530")
        self.collection_name = "climate_articles"
        # We use /app/data because that is where the volume should be mounted
        self.data_dir = "/app/data" 

    def get_client(self):
        return MilvusClient(uri=self.uri)

    def get_count(self):
        try:
            client = self.get_client()
            if not client.has_collection(self.collection_name):
                return 0
            # Explicitly load to ensure we get a real count
            client.load_collection(self.collection_name)
            res = client.query(collection_name=self.collection_name, filter="", output_fields=["count(*)"])
            return res[0]["count(*)"]
        except Exception as e:
            print(f"DB Connection Check Failed: {e}")
            return -1

    def ingest_default_data(self):
        parquet_path = os.path.join(self.data_dir, "climate_news_data.parquet")
        npy_path = os.path.join(self.data_dir, "climate_vectors.npy")

        # DEBUG PRINTS: This helps us find the path mismatch
        print(f"🔍 Checking for Parquet at: {os.path.abspath(parquet_path)}")
        print(f"🔍 Checking for NPY at: {os.path.abspath(npy_path)}")

        if not os.path.exists(parquet_path) or not os.path.exists(npy_path):
            return False, f"Files not found. Searched in: {self.data_dir}"

        client = self.get_client()
        
        # 1. Recreate Collection with specific primary key naming for LangChain
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)
        
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        # Use 'id' to match your previous schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2560)
        
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
            # Handle date formatting safely
            try:
                date_val = row.get('date', "")
                date_str = date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)
            except:
                date_str = ""

            entry = {
                "vector": vectors[idx].tolist(),
                "text": str(row.get('body', '')),
                "title": str(row.get('title', 'Untitled')),
                "category": str(row.get('category', 'News')),
                "date": date_str,
                "tags": str(row.get('tags', '[]'))
            }
            data_to_insert.append(entry)

        # 3. Insert and Load
        batch_size = 200
        for i in range(0, len(data_to_insert), batch_size):
            client.insert(collection_name=self.collection_name, data=data_to_insert[i:i+batch_size])
            
        client.load_collection(self.collection_name)
        return True, f"Successfully inserted {len(data_to_insert)} records."