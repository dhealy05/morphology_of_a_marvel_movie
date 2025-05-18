import pymupdf
#
from .section import *
from .section_collection import *
#
from analysis.distances import *
from analysis.tsne_animated import *
#
from embeddings.text_embedder import TextEmbedder

class Narrative(SectionCollection):

    def __init__(self, filepath, num_sections=5, perplexity=4, title=""):
        self.title = title
        self.source = filepath
        self.sections = self.split_pdf(self.get_data(filepath), num_sections)
        # animated functions
        self.reference_dict = {section.text: section.embedding for section in self.sections}
        self.visualizer = AnimatedTSNEVisualizer()
        self.visualizer.from_dict(self.reference_dict, perplexity=perplexity)

####### Embedding Analysis Functions

    def plot_similarities(self, output_path=None):
        """
        Create a line plot showing cosine similarities of each section to the average embedding.

        Args:
            output_path (str, optional): Path to save the plot. If None, the plot is displayed.

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        similarities = self.section_similarities_to_average()
        avg_similarity = np.mean(similarities)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the similarities
        section_indices = list(range(1, len(similarities) + 1))
        ax.plot(section_indices, similarities, 'o-', linewidth=2, markersize=8, label='Cosine Similarity')

        # Add the average line
        ax.axhline(y=avg_similarity, color='red', linestyle='--', label=f'Average ({avg_similarity:.3f})')

        # Set labels and title
        ax.set_xlabel('Section Index')
        ax.set_ylabel('Cosine Similarity to Average Embedding')
        ax.set_title('Section Embedding Similarities to Average Embedding')

        # Set x-axis ticks to be integers
        ax.set_xticks(section_indices)

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend()

        # Tight layout
        plt.tight_layout()

        # Save or show the plot
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

        return fig

####### Simple Progression Functions

    def animate(self):
        self.visualizer.create_animation('img/' + self.title + '_simple_animation.gif', show_line=True)

    def animate_distance(self):
        self.visualizer.create_distance_animation('img/' + self.title + '_distance.gif', metric="cosine", normalization="maxunit")

    def animate_combined(self):
        self.visualizer.create_combined_animation('img/' + self.title + '_dual_animation.gif', show_line=True)

    def plot_distances(self):
        # get distances
        distances_for_text = self.visualizer.calculate_consecutive_distances(metric="cosine", normalization="maxunit")
        # parse distances
        distances_parsed = [distance_for_text["distance"] for distance_for_text in distances_for_text]
        #
        plot_normalized_sequences([distances_parsed], show_individual=True)

####### Parse Source

    def get_data(self, filepath):
        #
        doc = pymupdf.open(filepath)
        #
        pdf_string = ""
        #
        for page in doc:
            text = page.get_text()
            pdf_string += text
        #
        return pdf_string

    def split_pdf(self, pdf_string, num_sections):
        """
        Split a PDF string into approximately equal sections.

        Parameters:
        pdf_string (str): The full text extracted from a PDF
        num_sections (int): Number of sections to split the text into

        Returns:
        list: A list of strings, each containing a section of the PDF text
        """
        # Calculate total length and section size
        total_length = len(pdf_string)
        section_size = total_length // num_sections

        sections = []
        start = 0
        section_count = 0

        for i in range(num_sections - 1):
            # Find the nearest paragraph or line break after the calculated position
            end = start + section_size

            # Adjust end position to avoid cutting in the middle of a paragraph
            if end < total_length:
                # Look for the next paragraph break (\n\n) or at least a line break (\n)
                next_para = pdf_string.find('\n\n', end)
                next_line = pdf_string.find('\n', end)

                if next_para != -1 and next_para - end < section_size // 2:
                    # If a paragraph break is within half a section size, use it
                    end = next_para + 2
                elif next_line != -1 and next_line - end < section_size // 4:
                    # If a line break is within quarter a section size, use it
                    end = next_line + 1

            # Add the section to our list
            section_text = pdf_string[start:end]
            section = Section(section_text, self.source, section_count, TextEmbedder().get_embedding(section_text), {})
            sections.append(section)
            section_count += 1
            start = end

        # Add the last section (from the last start position to the end)
        section_text = pdf_string[start:]
        section = Section(section_text, self.source, section_count, TextEmbedder().get_embedding(section_text), {})
        sections.append(section)
        section_count += 1

        return sections
