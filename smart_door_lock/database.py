"""
Database management untuk Smart Door Lock - SQLite
Python 3.9.6 compatible
"""
import sqlite3
import numpy as np
import os
from config import DATABASE_PATH


class FaceDatabase:
    """SQLite database untuk face embeddings"""
    
    def __init__(self, db_path=DATABASE_PATH):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Create table jika belum ada"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def add_user(self, user_id, embedding):
        """Add or update user embedding"""
        if isinstance(embedding, np.ndarray):
            embedding_bytes = embedding.astype(np.float32).tobytes()
        else:
            embedding_bytes = embedding
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO face_embeddings (user_id, embedding)
                VALUES (?, ?)
            ''', (user_id, embedding_bytes))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding user {user_id}: {e}")
            return False
    
    def get_user(self, user_id):
        """Get user embedding"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT embedding FROM face_embeddings WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                embedding = np.frombuffer(result[0], dtype=np.float32)
                return embedding
            return None
        except Exception as e:
            print(f"Error getting user {user_id}: {e}")
            return None
    
    def get_all_users(self):
        """Get all users and embeddings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT user_id, embedding FROM face_embeddings')
            results = cursor.fetchall()
            conn.close()
            
            database = {}
            for user_id, embedding_bytes in results:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                database[user_id] = embedding
            
            return database
        except Exception as e:
            print(f"Error getting all users: {e}")
            return {}
    
    def list_users(self):
        """List all users"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT user_id, created_at FROM face_embeddings ORDER BY created_at')
            results = cursor.fetchall()
            conn.close()
            
            return results
        except Exception as e:
            print(f"Error listing users: {e}")
            return []
    
    def delete_user(self, user_id):
        """Delete user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM face_embeddings WHERE user_id = ?', (user_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting user {user_id}: {e}")
            return False
    
    def user_exists(self, user_id):
        """Check if user exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM face_embeddings WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            return result is not None
        except Exception as e:
            print(f"Error checking user existence: {e}")
            return False
    
    def clear_database(self):
        """Delete all users"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM face_embeddings')
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False
    
    def get_user_count(self):
        """Get total number of users"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM face_embeddings')
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
        except Exception as e:
            print(f"Error getting user count: {e}")
            return 0
