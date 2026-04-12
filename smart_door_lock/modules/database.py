"""
Face Database Module - SQLite3 Version
- Use SQLite3 (built-in Python) untuk vector similarity search
- Store dan retrieve face embeddings
- FULLY compatible dengan Raspberry Pi 3 (NO PyArrow/LanceDB issues)
"""

import os
import sqlite3
import numpy as np
import json
from datetime import datetime
from config import DB_NAME, EMBEDDINGS_TABLE, DATA_DIR


class FaceDatabase:
    """Face database menggunakan SQLite3 - FULLY ARM Compatible (NO PyArrow issues!)"""
    
    def __init__(self, db_path=DB_NAME):
        """
        Initialize SQLite3 database
        
        Args:
            db_path: Path ke database SQLite3
        """
        self.db_path = db_path
        self.table_name = EMBEDDINGS_TABLE
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database schema
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite3 database dengan embedding table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table jika belum ada
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen TEXT
                )
            """)
            
            # Create index untuk search performance
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_name 
                ON {self.table_name}(name)
            """)
            
            conn.commit()
            conn.close()
            
            print(f"[INFO] SQLite3 Database initialized: {self.db_path}")
        except Exception as e:
            print(f"[ERROR] Database initialization failed: {e}")
            raise
    
    def create_table(self):
        """Compatibility method - already created in __init__"""
        self._init_db()
    
    def add_enrollment(self, name, embeddings, person_id=None):
        """
        Add enrollment dari multiple faces
        
        Args:
            name: Nama pengguna
            embeddings: List of embedding vectors (numpy arrays)
            person_id: ID untuk pengguna (auto-generate jika None)
            
        Returns:
            bool: Success status
        """
        if not embeddings or len(embeddings) == 0:
            print("[ERROR] No embeddings provided")
            return False
        
        try:
            # Generate person_id jika tidak ada
            if person_id is None:
                person_id = f"user_{int(datetime.now().timestamp())}"
            
            # Average semua embeddings
            if len(embeddings) == 1:
                final_embedding = embeddings[0]
            else:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                final_embedding = np.mean(embeddings_array, axis=0)
                # L2 normalize
                norm = np.linalg.norm(final_embedding)
                if norm > 0:
                    final_embedding = final_embedding / norm
            
            # Convert embedding ke binary
            embedding_bytes = final_embedding.astype(np.float32).tobytes()
            
            # Insert ke SQLite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute(f"""
                INSERT OR REPLACE INTO {self.table_name} 
                (id, name, embedding, created_at, last_seen)
                VALUES (?, ?, ?, ?, ?)
            """, (person_id, name, embedding_bytes, now, now))
            
            conn.commit()
            conn.close()
            
            print(f"[SUCCESS] Enrollment saved: {name} (ID: {person_id})")
            return True
            
        except Exception as e:
            print(f"[ERROR] Add enrollment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_all_users(self):
        """
        Get all users dari database
        
        Returns:
            list: List of user dicts {id, name, embedding}
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT id, name, embedding 
                FROM {self.table_name}
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            users = []
            for row in rows:
                user_id, name, embedding_bytes = row
                # Convert bytes back to numpy array
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                users.append({
                    'id': user_id,
                    'name': name,
                    'embedding': embedding
                })
            
            return users
            
        except Exception as e:
            print(f"[ERROR] Get all users failed: {e}")
            return []

    def list_user_names(self):
        """Return distinct enrolled user names."""
        users = self.get_all_users()
        names = sorted({u.get('name', 'UNKNOWN') for u in users})
        return names

    def get_user_embeddings_count(self, user_ref):
        """
        Count embeddings by user id or user name.

        Args:
            user_ref: user id string or user name string
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE id = ? OR name = ?",
                (user_ref, user_ref),
            )
            count = cursor.fetchone()[0]
            conn.close()
            return int(count)
        except Exception as e:
            print(f"[ERROR] Count user embeddings failed: {e}")
            return 0

    def search_similar(self, embedding, top_k=1, threshold=0.7):
        """
        Search most similar users using cosine similarity in Python.

        Args:
            embedding: Query embedding vector
            top_k: number of results
            threshold: minimum similarity
        """
        query = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        users = self.get_all_users()
        results = []
        for user in users:
            emb = user.get('embedding')
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32)
            if emb.shape != query.shape:
                continue

            denom = np.linalg.norm(emb) * query_norm
            if denom == 0:
                continue

            sim = float(np.dot(query, emb) / denom)
            sim = max(0.0, min(1.0, sim))
            if sim >= threshold:
                results.append(
                    {
                        'id': user.get('id'),
                        'name': user.get('name', 'UNKNOWN'),
                        'similarity': sim,
                    }
                )

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def delete_user(self, user_id):
        """
        Delete user dari database
        
        Args:
            user_id: User ID to delete
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"""
                DELETE FROM {self.table_name}
                WHERE id = ?
            """, (user_id,))
            
            conn.commit()
            conn.close()
            
            print(f"[INFO] User deleted: {user_id}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Delete user failed: {e}")
            return False

    def delete_user_by_name(self, name):
        """Delete all records for a user name."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.table_name} WHERE name = ?", (name,))
            conn.commit()
            conn.close()
            print(f"[INFO] User deleted by name: {name}")
            return True
        except Exception as e:
            print(f"[ERROR] Delete user by name failed: {e}")
            return False
    
    def get_stats(self):
        """
        Get database statistics
        
        Returns:
            dict: Stats {total_users, total_embeddings}
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            total_users = cursor.fetchone()[0]

            cursor.execute(f"SELECT DISTINCT name FROM {self.table_name} ORDER BY name")
            users = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                'total_users': total_users,
                'total_embeddings': total_users,
                'users': users,
                'db_path': self.db_path
            }
            
        except Exception as e:
            print(f"[ERROR] Get stats failed: {e}")
            return {'total_users': 0, 'total_embeddings': 0}
