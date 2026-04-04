"""
Face Tracking Module
- Track posisi wajah antar frame
- Ensure stable identification
"""

import numpy as np
from collections import defaultdict


class FaceTracker:
    """Simple face tracker menggunakan centroid tracking"""
    
    def __init__(self, max_distance=50, max_disappeared=30):
        """
        Initialize tracker
        
        Args:
            max_distance: Maximum distance untuk assign detection ke track
            max_disappeared: Frames dimana track bisa hilang sebelum dihapus
        """
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        
        # Track data
        self.tracks = {}  # {track_id: {"centroid": (x,y), "frames": N, "embeddings": []}}
        self.next_track_id = 0
        self.disappeared = defaultdict(int)
    
    def register_track(self, centroid, embedding=None):
        """
        Register new track
        
        Args:
            centroid: (x, y) centroid position
            embedding: Face embedding vector
            
        Returns:
            Track ID
        """
        track_id = self.next_track_id
        self.next_track_id += 1
        
        self.tracks[track_id] = {
            "centroid": centroid,
            "frames": 1,
            "embeddings": [embedding] if embedding is not None else []
        }
        self.disappeared[track_id] = 0
        
        return track_id
    
    def update_track(self, track_id, centroid, embedding=None):
        """
        Update existing track
        
        Args:
            track_id: Track ID to update
            centroid: New centroid position
            embedding: New face embedding
        """
        if track_id in self.tracks:
            self.tracks[track_id]["centroid"] = centroid
            self.tracks[track_id]["frames"] += 1
            
            if embedding is not None:
                self.tracks[track_id]["embeddings"].append(embedding)
            
            self.disappeared[track_id] = 0
    
    def deregister_track(self, track_id):
        """
        Deregister track
        
        Args:
            track_id: Track ID to remove
        """
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]
    
    def update(self, detections, embeddings=None):
        """
        Update tracker dengan detections baru
        
        Args:
            detections: List of (x, y, w, h, confidence)
            embeddings: Optional list of embeddings corresponding to detections
            
        Returns:
            Dict of {track_id: {"centroid": (x,y), "frames": N, "embedding": embedding}}
        """
        # Calculate centroids dari detections
        centroids = []
        for detection in detections:
            x, y, w, h, conf = detection
            cx = x + w // 2
            cy = y + h // 2
            centroids.append((cx, cy))
        
        # Handle case ketika tidak ada detection
        if len(centroids) == 0:
            # Mark all tracks as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister_track(track_id)
            return {}
        
        # Assign detections ke existing tracks
        if len(self.tracks) == 0:
            # Register new tracks
            for i, centroid in enumerate(centroids):
                embedding = embeddings[i] if embeddings is not None else None
                self.register_track(centroid, embedding)
        else:
            # Match detections dengan existing tracks
            track_ids = list(self.tracks.keys())
            
            # Calculate distance matrix
            distances = np.zeros((len(track_ids), len(centroids)))
            for i, track_id in enumerate(track_ids):
                track_centroid = self.tracks[track_id]["centroid"]
                for j, centroid in enumerate(centroids):
                    dx = track_centroid[0] - centroid[0]
                    dy = track_centroid[1] - centroid[1]
                    distances[i, j] = np.sqrt(dx*dx + dy*dy)
            
            # Greedy assignment
            used_detections = set()
            used_tracks = set()
            
            # Sort oleh distance
            for i, j in sorted(zip(*np.where(distances < self.max_distance)), 
                              key=lambda x: distances[x]):
                if i not in used_tracks and j not in used_detections:
                    track_id = track_ids[i]
                    embedding = embeddings[j] if embeddings is not None else None
                    self.update_track(track_id, centroids[j], embedding)
                    
                    used_tracks.add(i)
                    used_detections.add(j)
            
            # Register unmatched detections
            for j, centroid in enumerate(centroids):
                if j not in used_detections:
                    embedding = embeddings[j] if embeddings is not None else None
                    self.register_track(centroid, embedding)
            
            # Deregister unmatched tracks
            for i, track_id in enumerate(track_ids):
                if i not in used_tracks:
                    self.disappeared[track_id] += 1
                    if self.disappeared[track_id] > self.max_disappeared:
                        self.deregister_track(track_id)
        
        # Return current tracks
        result = {}
        for track_id, track_data in self.tracks.items():
            result[track_id] = {
                "centroid": track_data["centroid"],
                "frames": track_data["frames"],
                "embedding": track_data["embeddings"][-1] if track_data["embeddings"] else None
            }
        
        return result
    
    def get_stable_tracks(self, min_frames=5):
        """
        Get tracks yang sudah stable (ada selama min_frames)
        
        Args:
            min_frames: Minimum frames untuk dianggap stable
            
        Returns:
            Dict of stable tracks
        """
        stable = {}
        for track_id, track_data in self.tracks.items():
            if track_data["frames"] >= min_frames:
                stable[track_id] = track_data
        return stable
    
    def clear(self):
        """Clear semua tracks"""
        self.tracks = {}
        self.disappeared = defaultdict(int)
        self.next_track_id = 0
