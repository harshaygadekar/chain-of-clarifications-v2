"""
Result Caching System

Provides caching for agent chain results to speed up development and testing.
Uses file-based persistence with JSON storage.
"""

import json
import hashlib
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ResultCache:
    """
    File-based result cache for agent chain outputs.
    
    Features:
    - Hash-based key generation from question + context
    - JSON file persistence
    - Configurable TTL (time-to-live)
    - Memory cache with disk backup
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/results",
        ttl_hours: int = 24,
        enabled: bool = True
    ):
        """
        Initialize the result cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours (0 = no expiry)
            enabled: Whether caching is enabled
        """
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.enabled = enabled
        self.memory_cache: Dict[str, Dict] = {}
        self.hits = 0
        self.misses = 0
        
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Result cache initialized at {cache_dir}")
    
    def _generate_key(self, question: str, context: str, model: str = "") -> str:
        """Generate unique cache key from inputs."""
        content = f"{question}|{context}|{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key."""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _is_expired(self, cached_at: str) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_hours == 0:
            return False
        
        cached_time = datetime.fromisoformat(cached_at)
        expiry_time = cached_time + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def get(
        self,
        question: str,
        context: str,
        model: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available.
        
        Args:
            question: The question
            context: The context/document
            model: Model identifier (optional)
            
        Returns:
            Cached result dict or None if not found/expired
        """
        if not self.enabled:
            return None
        
        key = self._generate_key(question, context, model)
        
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not self._is_expired(entry['cached_at']):
                self.hits += 1
                logger.debug(f"Cache HIT (memory): {key[:8]}...")
                return entry['result']
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    entry = json.load(f)
                
                if not self._is_expired(entry['cached_at']):
                    # Populate memory cache
                    self.memory_cache[key] = entry
                    self.hits += 1
                    logger.debug(f"Cache HIT (disk): {key[:8]}...")
                    return entry['result']
                else:
                    # Expired, remove file
                    os.remove(cache_path)
            except (json.JSONDecodeError, KeyError):
                os.remove(cache_path)
        
        self.misses += 1
        return None
    
    def set(
        self,
        question: str,
        context: str,
        result: Dict[str, Any],
        model: str = ""
    ) -> None:
        """
        Cache a result.
        
        Args:
            question: The question
            context: The context/document
            result: Result dict to cache
            model: Model identifier (optional)
        """
        if not self.enabled:
            return
        
        key = self._generate_key(question, context, model)
        
        entry = {
            'question': question[:100],  # Truncate for readability
            'model': model,
            'cached_at': datetime.now().isoformat(),
            'result': result
        }
        
        # Save to memory
        self.memory_cache[key] = entry
        
        # Save to disk
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(entry, f, indent=2, default=str)
            logger.debug(f"Cached result: {key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")
    
    def clear(self) -> int:
        """
        Clear all cached results.
        
        Returns:
            Number of entries cleared
        """
        count = len(self.memory_cache)
        self.memory_cache.clear()
        
        if os.path.exists(self.cache_dir):
            for f in os.listdir(self.cache_dir):
                if f.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, f))
                    count += 1
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1%}",
            'memory_entries': len(self.memory_cache),
            'enabled': self.enabled
        }
