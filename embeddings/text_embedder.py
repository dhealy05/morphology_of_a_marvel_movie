import numpy as np
from typing import List, Optional, Union
from scipy.spatial.distance import cosine
#
from openai import OpenAI

class TextEmbedder:
    """A class to handle text embedding operations using OpenAI's API"""

    def __init__(self, model: str = "text-embedding-3-large"):
        """
        Initialize the TextEmbedder.

        Args:
            model (str): The OpenAI embedding model to use.
                        Defaults to "text-embedding-3-large".
        """
        self.client = OpenAI()
        self.model = model

    def get_reference_dictionary(self, texts: List[str]):
        reference_dict = {}
        for text in texts:
            embedding = self.get_embedding(text)
            if embedding is not None:
                reference_dict[text] = embedding
        return reference_dict

    def get_embedding(self,
                     text: Union[str, List[str]],
                     normalize: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get embeddings for one or more texts using OpenAI's API.

        Args:
            text (Union[str, List[str]]): Single text string or list of text strings
                                        to get embeddings for.
            normalize (bool): Whether to normalize the resulting vectors.
                            Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The embedding vector(s) as numpy array(s).
            For single text input, returns a single numpy array.
            For list input, returns a list of numpy arrays.

        Raises:
            Exception: If there's an error calling the OpenAI API.
        """
        # Convert single string to list for consistent processing
        input_texts = [text] if isinstance(text, str) else text

        try:
            print(f"Calling OpenAI API for {len(input_texts)} text embedding(s).")
            response = self.client.embeddings.create(
                model=self.model,
                input=input_texts
            )

            # Convert embeddings to numpy arrays
            embeddings = [np.array(data.embedding, dtype=np.float32)
                        for data in response.data]

            # Normalize if requested
            if normalize:
                embeddings = [self._normalize(emb) for emb in embeddings]

            # Return single array for single input, list for multiple inputs
            return embeddings[0] if isinstance(text, str) else embeddings

        except Exception as e:
            #raise Exception(f"Error creating embedding: {e}")
            print(f"Error creating embedding: {e}")
            return None

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Args:
            vector (np.ndarray): The vector to normalize.

        Returns:
            np.ndarray: The normalized vector.
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

######## Order by Similarity

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return 1 - cosine(a, b)  # cosine() returns distance, so 1 - distance = similarity

    def order_embeddings_by_similarity(self, embedding_dict):
        """
        Orders a dictionary of embeddings to maximize cosine similarity between neighbors

        Args:
            embedding_dict: Dictionary mapping keys to embedding vectors

        Returns:
            List of keys in optimal order
        """
        keys = list(embedding_dict.keys())
        vectors = list(embedding_dict.values())

        # If only one or zero elements, return them as is
        if len(keys) <= 1:
            return keys

        # Start with first element
        ordered_keys = [keys[0]]
        remaining_keys = keys[1:]
        remaining_vectors = vectors[1:]

        # Greedy algorithm: always pick the next item with highest similarity to current
        while remaining_keys:
            last_vector = embedding_dict[ordered_keys[-1]]

            # Find element with highest similarity to the last element in our chain
            similarities = [self.cosine_similarity(last_vector, vector) for vector in remaining_vectors]
            best_idx = np.argmax(similarities)

            # Add the best element to our ordered list
            ordered_keys.append(remaining_keys[best_idx])

            # Remove it from remaining elements
            remaining_keys.pop(best_idx)
            remaining_vectors.pop(best_idx)

        #return ordered_keys
        return {k: embedding_dict[k] for k in ordered_keys}
