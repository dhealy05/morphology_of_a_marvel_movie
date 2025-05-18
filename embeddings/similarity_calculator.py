import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, List
#
from narrative_structures.section import Section

class SimilarityCalculator:
    """
    A class to calculate cosine similarity between a Section's embedding and a reference dictionary.

    Attributes:
        section (Section): The section to compare against reference embeddings
        reference_dict (Dict[str, np.ndarray]): Dictionary of text->embedding pairs for comparison
    """

    def __init__(self, section: Section, reference_dict: Dict[str, np.ndarray]):
        """
        Initialize the SimilarityCalculator.

        Args:
            section (Section): The section to compare
            reference_dict (Dict[str, np.ndarray]): Dictionary of reference embeddings
        """
        self.section = section
        self.reference_dict = reference_dict

    def _cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1 (np.ndarray): First vector
            vector2 (np.ndarray): Second vector

        Returns:
            float: Cosine similarity value
        """
        dot_product = np.dot(vector1, vector2)
        norm_v1 = np.linalg.norm(vector1)
        norm_v2 = np.linalg.norm(vector2)

        # Prevent division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return dot_product / (norm_v1 * norm_v2)

######## Diffs

    def compute_diffs(self) -> Dict[str, np.ndarray]:
        """
        Compute the differences between the section's embedding and all reference embeddings.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping reference keys to difference vectors
        """
        differences = {}
        section_embedding = self.section.embedding

        # Convert to numpy array if it's a list
        if isinstance(section_embedding, list):
            section_embedding = np.array(section_embedding)

        for key, ref_embedding in self.reference_dict.items():
            # Compute element-wise difference
            diff = section_embedding - ref_embedding
            differences[key] = diff

        return differences

    def stack(self, dictionary: Dict[str, Any]) -> np.ndarray:
        # Ensure all values are numpy arrays of the same shape
        array_list = []
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                array_list.append(value)
            else:
                raise TypeError(f"Value for key '{key}' is not a numpy array.")

        return np.stack(array_list)

######### Similarities

    def compute_similarities(self, normalize: bool = True) -> Dict[str, float]:
        """
        Compute cosine similarity between the section's embedding and all reference embeddings.
        Args:
            normalize (bool): Whether to normalize similarity values to max / min 1 / 0
        Returns:
            Dict[str, float]: Dictionary mapping reference keys to similarity scores
        """
        similarities = {}
        section_embedding = self.section.embedding
        # Convert to numpy array if it's a list
        if isinstance(section_embedding, list):
            section_embedding = np.array(section_embedding)
        for key, ref_embedding in self.reference_dict.items():
            similarity = self._cosine_similarity(section_embedding, ref_embedding)
            similarities[key] = similarity

        # Add normalization logic
        if normalize and similarities:
            min_sim = min(similarities.values())
            max_sim = max(similarities.values())

            # Avoid division by zero if all similarities are the same
            if max_sim != min_sim:
                similarities = {
                    key: (sim - min_sim) / (max_sim - min_sim)
                    for key, sim in similarities.items()
                }
            else:
                # If all similarities are equal, set them all to 1.0 or 0.0
                # Here we choose 1.0 as it preserves the fact that they're equally similar
                similarities = {key: 1.0 for key in similarities}

        #similarities = {
        #    k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        #}

        return similarities

    def flatten(self, dictionary: Dict[str, Any]) -> np.ndarray:
        # Ensure all values are numpy arrays of the same shape
        array_list = []
        for key, value in dictionary.items():
            array_list.append(value)

        return np.array(array_list)

######## Weighted Concatenation

    def compute_similarity_weighted_embedding(self, method="weighted_average", top_k=5):
        """
        Compute a similarity-weighted embedding combining all reference embeddings.

        Args:
            method (str): Method to use - "weighted_average", "weighted_concat", or "top_k_weighted"
            top_k (int): Number of top embeddings to use if method is "top_k_weighted"

        Returns:
            np.ndarray: A single embedding representing the weighted relationship
        """
        # Get similarities
        similarities = self.compute_similarities(normalize=True)

        # Extract reference embeddings and their similarities in matching order
        ref_keys = list(similarities.keys())
        sim_values = np.array([similarities[key] for key in ref_keys])
        ref_embeddings = np.array([self.reference_dict[key] for key in ref_keys])

        # Ensure we have valid embeddings
        if len(ref_embeddings) == 0:
            return self.section.embedding

        # Compute weighted embedding based on selected method
        if method == "weighted_average":
            # Weighted average of all embeddings
            total_similarity = np.sum(sim_values)
            if total_similarity > 0:
                normalized_sims = sim_values / total_similarity
            else:
                normalized_sims = np.ones_like(sim_values) / len(sim_values)

            weighted_embedding = np.zeros_like(ref_embeddings[0])
            for i, embedding in enumerate(ref_embeddings):
                weighted_embedding += normalized_sims[i] * embedding

            return weighted_embedding

        elif method == "weighted_concat":
            # Reshape similarities for broadcasting
            weights = sim_values.reshape(-1, 1)
            # Weight each embedding
            weighted_embeddings = ref_embeddings * weights
            # Flatten all embeddings into one vector
            return weighted_embeddings.flatten()

        elif method == "top_k_weighted":
            # Only use top K embeddings
            k = min(top_k, len(sim_values))
            top_indices = np.argsort(sim_values)[-k:]
            top_similarities = sim_values[top_indices]

            # Re-normalize top similarities
            top_total = np.sum(top_similarities)
            if top_total > 0:
                top_normalized = top_similarities / top_total
            else:
                top_normalized = np.ones_like(top_similarities) / k

            weighted_embedding = np.zeros_like(ref_embeddings[0])
            for i, idx in enumerate(top_indices):
                weighted_embedding += top_normalized[i] * ref_embeddings[idx]

            return weighted_embedding

        else:
            raise ValueError(f"Unknown method: {method}")

####### Plot

    def plot_radio(self, similarities: Dict[str, float] = None, title: str = "Cosine Similarity Radar Plot",
                  figsize: tuple = (10, 10), save_path: str = "img/radio.png") -> None:
        """
        Creates a radar plot where each key in the dictionary is represented as a point,
        with the distance from the center representing the cosine similarity.

        Args:
            similarities (Dict[str, float], optional): Dictionary of similarity scores.
                If None, computes similarities using compute_similarities().
            title (str, optional): Title for the plot. Defaults to "Cosine Similarity Radar Plot".
            figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 10).
            save_path (str, optional): Path to save the figure. If None, the figure is displayed.

        Returns:
            None: Displays or saves the plot
        """
        if similarities is None:
            similarities = self.compute_similarities()

        # Number of categories (keys in the dictionary)
        N = len(similarities)
        if N < 3:
            raise ValueError("Need at least 3 items to create a meaningful radar plot")

        # Create a figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)

        # Compute angles for each category (in radians)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()

        # Make the plot circular by appending the first angle at the end
        angles += angles[:1]

        # Get the values and keys
        values = list(similarities.values())
        keys = list(similarities.keys())

        # Make values circular by appending the first value at the end
        values += values[:1]

        # Convert similarity to distance (1 = closest to center, 0 = farthest)
        # Optional: Invert values if you want highest similarity to be farthest from center
        # values = [1 - v for v in values]

        # Plot the radar
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        # Set category labels
        ax.set_xticks(angles[:-1])

        # show every 4th label so as not to overwhelm with text
        display_labels = [key if i % 4 == 0 else "" for i, key in enumerate(keys)]
        ax.set_xticklabels(display_labels)
        #ax.set_xticklabels(keys)

        # Set y-axis limits
        ax.set_ylim(0, 1)

        # Add title
        plt.title(title, size=15)

        # Adjust grid
        ax.grid(True)

        # Add similarity values as annotations
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            # Add value annotation
            ax.annotate(f"{value:.2f}",
                       xy=(angle, value),
                       xytext=(angle, value + 0.1),
                       ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

        plt.savefig(save_path, dpi=300, bbox_inches='tight')

###### Plot Heatmap

    def plot_similarity_heatmap(self, similarities_list, title="Similarity Heatmap",
                               cmap="coolwarm", figsize=(12, 10), show_values=False,
                               fontsize=8, x_labels=None, path_title=""):
        """
        Plot a heatmap of similarity scores with improved readability for large datasets,
        with axes flipped from the original implementation.

        Args:
            similarities_list (List[Dict[str, float]]): List of dictionaries containing similarity scores
            title (str): Title for the heatmap
            cmap (str): Colormap to use (default: coolwarm)
            figsize (tuple): Fixed figure size as (width, height)
            show_values (bool): Whether to annotate cells with values (not recommended for large datasets)
            fontsize (int): Size of font for labels
            x_labels (list): Optional custom labels for columns, if None will use "Item 1", "Item 2", etc.

        Returns:
            matplotlib.figure.Figure: The figure object containing the heatmap
        """
        # Get all unique keys across all dictionaries
        all_keys = list(similarities_list[0].keys())

        # Create a matrix for the heatmap - but TRANSPOSED from the original version
        matrix = np.zeros((len(all_keys), len(similarities_list)))

        # Fill the matrix with similarity values - note the transposed indices [j, i] instead of [i, j]
        for i, similarities in enumerate(similarities_list):
            for j, key in enumerate(all_keys):
                if key in similarities:
                    matrix[j, i] = similarities[key]  # Note the swap of indices here

        # Create a DataFrame for better visualization
        if x_labels is None:
            x_labels = [f"{i}" for i in range(len(similarities_list))]

        df = pd.DataFrame(matrix, index=all_keys, columns=x_labels)

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the heatmap
        sns.heatmap(
            df,
            annot=show_values,  # Turn off text annotations by default
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.5}  # Make colorbar smaller
        )

        # Set labels and title
        ax.set_title(title, fontsize=fontsize+6, pad=20)
        ax.set_ylabel("Reference Keys", fontsize=fontsize+2)  # Now y-axis has the keys
        ax.set_xlabel("Sections", fontsize=fontsize+2)  # Now x-axis has the items

        # Handle y-axis labels (now the keys)
        max_labels = 40  # Maximum number of labels to display

        if len(all_keys) > max_labels:
            # Show only a subset of labels
            step = len(all_keys) // max_labels + 1
            visible_positions = np.arange(0, len(all_keys), step)

            # Create empty labels list but put actual labels at visible positions
            sparse_labels = [""] * len(all_keys)
            for i in visible_positions:
                if i < len(all_keys):
                    sparse_labels[i] = all_keys[i]

            ax.set_yticks(np.arange(len(all_keys)) + 0.5)
            ax.set_yticklabels(sparse_labels, fontsize=fontsize)
        else:
            plt.yticks(fontsize=fontsize)

        # Handle column labels (x-axis, now the items)
        if len(x_labels) > max_labels:
            # Show only a subset of labels
            step = len(x_labels) // max_labels + 1
            visible_positions = np.arange(0, len(x_labels), step)

            # Create empty labels list but put actual labels at visible positions
            sparse_labels = [""] * len(x_labels)
            for i in visible_positions:
                if i < len(x_labels):
                    sparse_labels[i] = x_labels[i]

            plt.xticks(np.arange(len(x_labels)) + 0.5, sparse_labels, fontsize=fontsize)
        else:
            plt.xticks(fontsize=fontsize, rotation=45, ha='right')

        # Add grid to help with readability
        ax.grid(False)

        # Adjust layout to fit in the fixed size
        plt.tight_layout()
        plt.savefig("img/" + path_title + "_heatmap.png")
        return fig
