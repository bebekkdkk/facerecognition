"""
Face Database Module
- LanceDB untuk vector similarity search
- Store dan retrieve face embeddings
"""

import os
import lancedb
import numpy as np
import json
import pyarrow as pa
from datetime import datetime
from config import DB_NAME, EMBEDDINGS_TABLE, DATA_DIR


class FaceDatabase:
    """Face database dengan LanceDB untuk vector similarity search"""
    
    def __init__(self, db_path=DB_NAME):
        """
        Initialize database
        
        Args:
            db_path: Path ke database
        """
        self.db_path = db_path
        self.db = None
        self.table_name = EMBEDDINGS_TABLE
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._connect()
    
    def _connect(self):
        """Connect ke LanceDB"""
        try:
            self.db = lancedb.connect(self.db_path)
            print(f"[INFO] Database connected: {self.db_path}")
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")
            raise
    
    def create_table(self):
        """Create embedding table jika belum ada"""
        try:
            if self.table_name in self.db.table_names():
                print(f"[INFO] Table '{self.table_name}' already exists")
            else:
                # Create schema
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("name", pa.string()),
                    pa.field("embedding", pa.list_(pa.float32(), 512)),
                    pa.field("timestamp", pa.string())
                ])
                
                # Create dummy data dengan schema
                data = {
                    "id": ["dummy"],
                    "name": ["dummy"],
                    "embedding": [np.zeros(512, dtype=np.float32)],
                    "timestamp": [datetime.now().isoformat()]
                }
                
                # Create table
                table = pa.Table.from_pydict(data, schema=schema)
                self.db.create_table(self.table_name, data=table)
                
                # Delete dummy data
                self.db.open_table(self.table_name).delete("id = 'dummy'")
                print(f"[INFO] Table '{self.table_name}' created")
        except Exception as e:
            print(f"[WARNING] Table creation: {e}")
    
    def add_enrollment(self, name, embeddings, person_id=None):
        """
        Add enrollment dari multiple faces
        
        Args:
            name: Nama pengguna
            embeddings: List of embedding vectors
            person_id: ID untuk pengguna (auto-generate jika None)
            
        Returns:
            Success status
        """
        try:
            if person_id is None:
                person_id = f"{name}_{int(datetime.now().timestamp())}"
            
            table = self.db.open_table(self.table_name)
            
            # Add each embedding
            for i, embedding in enumerate(embeddings):
                data = {
                    "id": f"{person_id}_{i}",
                    "name": name,
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "timestamp": datetime.now().isoformat()
                }
                table.add([data])
            
            print(f"[INFO] Added {len(embeddings)} embeddings for user: {name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Enrollment add failed: {e}")
            return False
    
    def search_similar(self, query_embedding, top_k=1, threshold=0.6):
        """
        Search untuk wajah yang mirip
        
        Args:
            query_embedding: Query embedding vector
            top_k: Jumlah hasil teratas
            threshold: Minimum similarity threshold
            
        Returns:
            List of results [(name, similarity_score, timestamp), ...]
        """
        try:
            table = self.db.open_table(self.table_name)
            
            # Convert ke list jika numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Vector search
            results = table.search(query_embedding).limit(top_k).to_list()
            
            # Filter oleh threshold dan format output
            matches = []
            for result in results:
                # Hitung similarity dari distance
                # LanceDB uses L2 distance
                distance = result.get('_distance', float('inf'))
                # Convert distance ke similarity (0-1)
                similarity = 1.0 / (1.0 + distance)
                
                if similarity >= threshold:
                    matches.append({
                        "name": result.get('name', 'Unknown'),
                        "similarity": similarity,
                        "timestamp": result.get('timestamp', 'N/A'),
                        "id": result.get('id', 'N/A')
                    })
            
            return matches
            
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []
    
    def get_all_users(self):
        """
        Get semua unique users dalam database
        
        Returns:
            List of user names
        """
        try:
            table = self.db.open_table(self.table_name)
            results = table.search().limit(10000).to_list()
            
            # Get unique names
            unique_names = list(set([r.get('name', 'Unknown') for r in results]))
            return unique_names
            
        except Exception as e:
            print(f"[ERROR] Get users failed: {e}")
            return []
    
    def delete_user(self, name):
        """
        Delete semua embedding untuk user tertentu
        
        Args:
            name: Nama user
            
        Returns:
            Success status
        """
        try:
            table = self.db.open_table(self.table_name)
            # LanceDB delete dengan where clause
            table.delete(f"name = '{name}'")
            print(f"[INFO] Deleted user: {name}")
            return True
        except Exception as e:
            print(f"[ERROR] Delete user failed: {e}")
            return False
    
    def get_user_embeddings_count(self, name):
        """
        Get jumlah embeddings untuk user
        
        Args:
            name: Nama user
            
        Returns:
            Count
        """
        try:
            table = self.db.open_table(self.table_name)
            results = table.search().limit(10000).to_list()
            count = sum(1 for r in results if r.get('name') == name)
            return count
        except Exception as e:
            print(f"[ERROR] Count embeddings failed: {e}")
            return 0
    
    def get_stats(self):
        """
        Get database statistics
        
        Returns:
            Dict dengan stats
        """
        try:
            table = self.db.open_table(self.table_name)
            results = table.search().limit(10000).to_list()
            
            unique_users = set(r.get('name') for r in results)
            
            stats = {
                "total_embeddings": len(results),
                "total_users": len(unique_users),
                "users": list(unique_users)
            }
            
            return stats
        except Exception as e:
            print(f"[ERROR] Get stats failed: {e}")
            return {}
    
    def export_metadata(self, output_path):
        """
        Export metadata ke JSON
        
        Args:
            output_path: Output file path
        """
        try:
            stats = self.get_stats()
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"[INFO] Metadata exported to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
