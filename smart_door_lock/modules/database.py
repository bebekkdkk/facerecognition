"""
Face Database Module
- LanceDB untuk vector similarity search
- Store dan retrieve face embeddings
- Optimized untuk Raspberry Pi
"""

import os
import lancedb
import numpy as np
import json
import pyarrow as pa
from datetime import datetime
from config import DB_NAME, EMBEDDINGS_TABLE, DATA_DIR


class FaceDatabase:
    """Face database dengan LanceDB untuk vector similarity search - RPi optimized"""
    
    def __init__(self, db_path=DB_NAME):
        """
        Initialize database
        
        Args:
            db_path: Path ke database
        """
        self.db_path = db_path
        self.db = None
        self.table_name = EMBEDDINGS_TABLE
        self.connection_retries = 3
        self.retry_delay = 0.1  # seconds
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._connect()
    
    def _connect(self):
        """Connect ke LanceDB dengan retry logic untuk ARM"""
        last_error = None
        for attempt in range(self.connection_retries):
            try:
                self.db = lancedb.connect(self.db_path)
                print(f"[INFO] Database connected: {self.db_path}")
                return
            except Exception as e:
                last_error = e
                if attempt < self.connection_retries - 1:
                    import time
                    print(f"[WARNING] Connection attempt {attempt + 1} failed, retrying...")
                    time.sleep(self.retry_delay)
        
        print(f"[ERROR] Database connection failed after {self.connection_retries} attempts: {last_error}")
        raise last_error
    
    def create_table(self):
        """Create embedding table jika belum ada - dengan error handling untuk RPi"""
        try:
            if self.db is None:
                print("[ERROR] Database not connected")
                return
            
            if self.table_name in self.db.table_names():
                print(f"[INFO] Table '{self.table_name}' already exists")
                return
            
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
            if self.db is None:
                print("[ERROR] Database not connected")
                return False
            
            if person_id is None:
                person_id = f"{name}_{int(datetime.now().timestamp())}"
            
            table = self.db.open_table(self.table_name)
            
            # Add each embedding dengan batch processing untuk RPi efficiency
            batch_size = 10
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size]
                for j, embedding in enumerate(batch):
                    data = {
                        "id": f"{person_id}_{i+j}",
                        "name": name,
                        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                        "timestamp": datetime.now().isoformat()
                    }
                    table.add([data])
            
            print(f"[INFO] Added {len(embeddings)} embeddings for user: {name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Enrollment add failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_similar(self, query_embedding, top_k=1, threshold=0.6):
        """
        Search untuk wajah yang mirip dengan error handling untuk ARM
        
        Args:
            query_embedding: Query embedding vector
            top_k: Jumlah hasil teratas
            threshold: Minimum similarity threshold
            
        Returns:
            List of results dengan format yang consistent
        """
        try:
            if self.db is None:
                print("[ERROR] Database not connected")
                return []
            
            table = self.db.open_table(self.table_name)
            
            # Convert ke list jika numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Validate embedding
            if not query_embedding or len(query_embedding) != 512:
                print(f"[WARNING] Invalid embedding dimension: {len(query_embedding) if query_embedding else 0}")
                return []
            
            # Vector search dengan timeout handling
            results = table.search(query_embedding).limit(top_k).to_list()
            
            # Filter oleh threshold dan format output
            matches = []
            for result in results:
                try:
                    # Hitung similarity dari distance
                    distance = result.get('_distance', float('inf'))
                    
                    # Handle NaN atau infinite distances
                    if np.isnan(distance) or np.isinf(distance):
                        continue
                    
                    # Convert distance ke similarity (0-1)
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= threshold:
                        matches.append({
                            "name": result.get('name', 'Unknown'),
                            "similarity": float(similarity),
                            "timestamp": result.get('timestamp', 'N/A'),
                            "id": result.get('id', 'N/A')
                        })
                except Exception as e:
                    print(f"[WARNING] Error processing search result: {e}")
                    continue
            
            return matches
            
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_all_users(self):
        """
        Get semua unique users dalam database
        
        Returns:
            List of user names
        """
        try:
            if self.db is None:
                print("[ERROR] Database not connected")
                return []
            
            table = self.db.open_table(self.table_name)
            results = table.search().limit(10000).to_list()
            
            # Get unique names dengan set untuk efficiency
            unique_names = list(set([r.get('name', 'Unknown') for r in results]))
            return sorted(unique_names)
            
        except Exception as e:
            print(f"[ERROR] Get users failed: {e}")
            return []
    
    def delete_user(self, name):
        """
        Delete semua embedding untuk user tertentu dengan proper SQL escaping
        
        Args:
            name: Nama user
            
        Returns:
            Success status
        """
        try:
            if self.db is None:
                print("[ERROR] Database not connected")
                return False
            
            table = self.db.open_table(self.table_name)
            # Escape name untuk SQL injection prevention
            escaped_name = name.replace("'", "''")
            table.delete(f"name = '{escaped_name}'")
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
            if self.db is None:
                print("[ERROR] Database not connected")
                return 0
            
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
            if self.db is None:
                print("[ERROR] Database not connected")
                return {"total_users": 0, "total_embeddings": 0, "users": []}
            
            table = self.db.open_table(self.table_name)
            results = table.search().limit(10000).to_list()
            
            if not results:
                return {"total_users": 0, "total_embeddings": 0, "users": []}
            
            unique_users = set(r.get('name') for r in results if r.get('name'))
            
            stats = {
                "total_embeddings": len(results),
                "total_users": len(unique_users),
                "users": sorted(list(unique_users))
            }
            
            return stats
        except Exception as e:
            print(f"[ERROR] Get stats failed: {e}")
            return {"total_users": 0, "total_embeddings": 0, "users": []}
    
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
