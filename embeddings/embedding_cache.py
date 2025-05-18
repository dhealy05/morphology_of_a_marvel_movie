import os
import json
import pickle
import hashlib
import numpy as np
from typing import Dict, List, Optional, Union, Any
#
from .text_embedder import TextEmbedder

class EmbeddingCache:
    """A class to handle caching of text embeddings"""

    def __init__(self, cache_dir: str = "embedding_cache"):
        """
        Initialize the EmbeddingCache.

        Args:
            cache_dir (str): Directory to store cache files. Will be created if it doesn't exist.
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.embedder = TextEmbedder()

    def get_embedding(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Get embedding for a text, using cache if available.

        Args:
            text (str): The text to get embedding for
            use_cache (bool): Whether to use cached embedding if available

        Returns:
            Optional[np.ndarray]: The embedding vector as numpy array or None if failed
        """
        if use_cache:
            # Try to get from cache first
            cached_embedding = self._get_from_cache(text)
            if cached_embedding is not None:
                return cached_embedding

        # Generate new embedding
        embedding = self.embedder.get_embedding(text)

        # Cache the new embedding
        if embedding is not None and use_cache:
            self._add_to_cache(text, embedding)

        return embedding

    def get_reference_dictionary(self, texts: List[str], use_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        Get a dictionary of texts and their embeddings, using cache when possible.

        Args:
            texts (List[str]): List of texts to get embeddings for
            use_cache (bool): Whether to use cached embeddings if available

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping texts to their embeddings
        """
        reference_dict = {}

        # First check which texts we already have in cache
        uncached_texts = []
        if use_cache:
            for text in texts:
                cached_embedding = self._get_from_cache(text)
                if cached_embedding is not None:
                    reference_dict[text] = cached_embedding
                else:
                    uncached_texts.append(text)
        else:
            uncached_texts = texts

        # Get embeddings for texts not in cache
        if uncached_texts:
            embeddings = self.embedder.get_embedding(uncached_texts)

            # Handle case where a single embedding is returned
            if len(uncached_texts) == 1 and not isinstance(embeddings, list):
                embeddings = [embeddings]

            # Add new embeddings to reference dict and cache
            if embeddings:
                for text, embedding in zip(uncached_texts, embeddings):
                    if embedding is not None:
                        reference_dict[text] = embedding
                        if use_cache:
                            self._add_to_cache(text, embedding)

        return reference_dict

    def _get_cache_key(self, text: str) -> str:
        """Generate a safe filename from text to use as cache key"""
        # Simple hash function to create a filename-safe representation
        return hashlib.md5(text.encode()).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Try to retrieve embedding from cache"""
        cache_key = self._get_cache_key(text)
        pickle_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading from cache: {e}")

        return None

    def _add_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text)
        pickle_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        json_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        # Save as pickle (preserves numpy arrays directly)
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"Error saving to pickle cache: {e}")

        # Save as JSON (convert numpy array to list)
        try:
            with open(json_path, 'w') as f:
                json.dump(embedding.tolist(), f)
        except Exception as e:
            print(f"Error saving to JSON cache: {e}")

    def save_reference_dict(self, reference_dict: Dict[str, np.ndarray], filename: str = "reference_embeddings") -> None:
        """
        Save an entire reference dictionary to disk.

        Args:
            reference_dict (Dict[str, np.ndarray]): Dictionary of text->embedding pairs
            filename (str): Base filename to save to (without extension)
        """
        # Save as pickle
        pickle_path = os.path.join(self.cache_dir, f"{filename}.pkl")
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(reference_dict, f)
            print(f"Reference dictionary saved to {pickle_path}")
        except Exception as e:
            print(f"Error saving reference dictionary to pickle: {e}")

        # Save as JSON
        json_path = os.path.join(self.cache_dir, f"{filename}.json")
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_dict = {text: emb.tolist() for text, emb in reference_dict.items()}
            with open(json_path, 'w') as f:
                json.dump(json_dict, f)
            print(f"Reference dictionary saved to {json_path}")
        except Exception as e:
            print(f"Error saving reference dictionary to JSON: {e}")

    def load_reference_dict(self, filename: str = "reference_embeddings") -> Dict[str, np.ndarray]:
        """
        Load a reference dictionary from disk.

        Args:
            filename (str): Base filename to load from (without extension)

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping texts to their embeddings
        """
        pickle_path = os.path.join(self.cache_dir, f"{filename}.pkl")

        # Try loading from pickle first (preserves numpy arrays)
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading reference dictionary from pickle: {e}")

        # Fall back to JSON if pickle fails or doesn't exist
        json_path = os.path.join(self.cache_dir, f"{filename}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    json_dict = json.load(f)
                # Convert lists back to numpy arrays
                ref_dict = {text: np.array(emb, dtype=np.float32) for text, emb in json_dict.items()}
                return ref_dict
            except Exception as e:
                print(f"Error loading reference dictionary from JSON: {e}")

        print(f"No cache file found at {pickle_path} or {json_path}")
        return {}
