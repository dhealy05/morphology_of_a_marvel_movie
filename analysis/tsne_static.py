import numpy as np
from typing import List
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
#
from narrative_structures.section import Section

def plot_section_embeddings(sections: List[Section], perplexity: int = 5, random_state: int = 42,
                           figsize: tuple = (10, 8), title: str = "t-SNE Plot of Section Embeddings",
                           show_labels: bool = False, label_max_length: int = 30,
                           show_line: bool = False, line_color: str = 'gray',
                           line_style: str = '--', alpha: float = 0.6, path_title=""):
    """
    Create a t-SNE plot of embeddings for a list of Section objects.
    Sections are colored by their relative position in their narrative and
    shaped by which narrative they belong to.

    Args:
        sections (List[Section]): List of Section objects to plot (with pre-populated embedding property)
        perplexity (int): t-SNE perplexity parameter (default: 5)
        random_state (int): Random seed for reproducibility (default: 42)
        figsize (tuple): Figure size as (width, height) (default: (10, 8))
        title (str): Plot title (default: "t-SNE Plot of Section Embeddings")
        show_labels (bool): Whether to show section labels (default: False)
        label_max_length (int): Maximum length for displayed labels (default: 30)
        show_line (bool): Whether to show lines connecting consecutive sections (default: True)
        line_color (str): Color of connecting lines (default: 'gray')
        line_style (str): Style of connecting lines (default: '--')
        alpha (float): Transparency of connecting lines (default: 0.6)

    Returns:
        tuple: (plt.Figure, plt.Axes) - The created figure and axes objects
    """
    # Get embeddings from sections
    embeddings = [section.embedding for section in sections]

    # Check if embeddings exist
    if not embeddings or any(emb is None for emb in embeddings):
        raise ValueError("One or more sections have missing embeddings. Ensure all sections have the 'embedding' property populated.")

    # Convert to numpy array
    embeddings_array = np.array(embeddings)

    # Adjust perplexity if necessary
    # t-SNE requires perplexity to be less than n-1
    effective_perplexity = min(perplexity, len(embeddings) - 1)
    if effective_perplexity < perplexity:
        print(f"Warning: Perplexity adjusted from {perplexity} to {effective_perplexity} to match data size")

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=effective_perplexity,
                random_state=random_state, max_iter=1000)
    tsne_results = tsne.fit_transform(embeddings_array)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Group sections by narrative_id
    narrative_groups = {}
    for section in sections:
        if section.narrative_id not in narrative_groups:
            narrative_groups[section.narrative_id] = []
        narrative_groups[section.narrative_id].append(section)

    # Define markers for different narratives
    markers = ['o', '^', 's', 'D', 'v', 'p', '*', 'X', 'h', 'H', 'd', '>', '<', '+', 'x', '.', ',', '|', '_']

    # Extract simplified narrative names from paths (e.g., "guardians-of-the-galaxy-vol-2" from path)
    def extract_movie_name(path):
        # Split the path and get the filename without extension
        import os
        basename = os.path.basename(path)
        movie_name = os.path.splitext(basename)[0]
        return movie_name

    # Plot each narrative group with its own marker
    legend_handles = []

    for i, (narrative_id, group) in enumerate(narrative_groups.items()):
        marker = markers[i % len(markers)]
        movie_name = extract_movie_name(narrative_id)

        # Sort sections by their index
        group.sort(key=lambda x: x.section_index)

        # Get section indices
        indices = [s.section_index for s in group]

        # Calculate normalized positions (0 to 1) based on section's position in its narrative
        num_sections = len(group)
        normalized_positions = [idx / (num_sections - 1) if num_sections > 1 else 0.5 for idx in indices]

        # Get the indices of these sections in the original sections list
        section_indices = [sections.index(section) for section in group]

        # Plot points with color based on normalized position in narrative
        for j, section_idx in enumerate(section_indices):
            ax.scatter(tsne_results[section_idx, 0], tsne_results[section_idx, 1],
                      c=[normalized_positions[j]], cmap='viridis',
                      marker=marker, s=100, alpha=0.8, vmin=0, vmax=1)

        # Create lines connecting sections within this narrative if requested
        if show_line and len(group) > 1:
            # Sort section_indices by the original section_index to ensure correct sequence
            sorted_indices = sorted(range(len(group)), key=lambda k: group[k].section_index)
            sorted_section_indices = [section_indices[i] for i in sorted_indices]

            for j in range(len(sorted_section_indices) - 1):
                ax.plot([tsne_results[sorted_section_indices[j], 0],
                         tsne_results[sorted_section_indices[j + 1], 0]],
                        [tsne_results[sorted_section_indices[j], 1],
                         tsne_results[sorted_section_indices[j + 1], 1]],
                        color=line_color, linestyle=line_style, alpha=alpha)

        # Add to legend
        legend_handle = mpatches.Patch(color='gray', label=f"{movie_name}")
        legend_handles.append((legend_handle, marker))

    # Add labels if requested
    if show_labels:
        for i, section in enumerate(sections):
            short_label = section.get_display_label(label_max_length)
            ax.annotate(short_label,
                        (tsne_results[i, 0], tsne_results[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

    # Create custom legend with markers
    from matplotlib.lines import Line2D
    custom_lines = []
    for handle, marker in legend_handles:
        custom_lines.append(Line2D([0], [0], marker=marker, color='gray',
                                  markersize=10, markerfacecolor='gray',
                                  linestyle='None', label=handle.get_label()))

    # Add colorbar to show progression through narrative
    import matplotlib as mpl
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Relative Position in Narrative (0-100%)')

    # Add legend for narratives/movies
    ax.legend(handles=custom_lines, loc='best', title="Movies")

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("img/" + path_title + "static_tsne.png")
    return fig, ax
