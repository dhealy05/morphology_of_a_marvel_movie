import math
import hashlib
import numpy as np
import pandas as pd
import seaborn as sns
import bar_chart_race as bcr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
#
from embeddings.similarity_calculator import SimilarityCalculator

class SectionCollection:

    def __init__(self, source, sections, perplexity=5):
      self.source = source
      self.sections = sections
      self.title = source

######## Section Winners

    def get_section_winners(self, prune_references=False):
        """
        Analyzes each section to find which reference text has the highest similarity.

        Args:
            prune_references (bool): If True, updates self.reference_dict to only keep winning references
                                    in the order they first appear as winners

        Returns:
            dict: Dictionary with winner names as keys and counts as values
        """
        # Dictionary to store counts of winners
        win_counts = {}
        # List to track unique winners in order of first appearance
        ordered_winners = []

        # Analyze each section
        for section in self.sections:
            # Create similarity calculator for this section
            calc = SimilarityCalculator(section, self.reference_dict)
            # Get similarities and find the winner
            similarities = calc.compute_similarities(normalize=True)
            winner = max(similarities.items(), key=lambda x: x[1])[0]
            # Update win counts
            win_counts[winner] = win_counts.get(winner, 0) + 1
            # Add to ordered list of unique winners (if not already there)
            if winner not in ordered_winners:
                ordered_winners.append(winner)

        # Optionally prune the reference dictionary to only include winners
        if prune_references and ordered_winners:
            # Create a new dictionary with only the winning references, preserving order
            pruned_refs = {k: self.reference_dict[k] for k in ordered_winners}
            # Update the reference dictionary
            self.reference_dict = pruned_refs

        return win_counts

    def create_winners_pie_chart(self, win_counts, internal_threshold=4.0, external_threshold=1.9):
        """
        Creates a pie chart with:
        - Internal labels for large sections
        - External labels for medium sections
        - Grouped 'Other' for very small sections

        Args:
            win_counts (dict): Category name -> count
            internal_threshold (float): % cutoff for internal labels
            external_threshold (float): % cutoff for external labels

        Returns:
            matplotlib.figure.Figure
        """

        total_sections = sum(win_counts.values())
        percentages = {k: (v / total_sections) * 100 for k, v in win_counts.items()}
        sorted_items = sorted(percentages.items(), key=lambda x: (-x[1], x[0]))

        # Group values by threshold
        internal_items = []
        external_items = []
        other_total = 0.0

        for label, pct in sorted_items:
            if pct >= internal_threshold:
                internal_items.append((label, pct))
            elif pct >= external_threshold:
                external_items.append((label, pct))
            else:
                other_total += pct

        if other_total > 0:
            external_items.append(("Other", other_total))

        final_items = internal_items + external_items

        # Generate consistent colors
        colors = []
        for label, _ in final_items:
            hash_object = hashlib.md5(label.encode())
            hex_color = '#' + hash_object.hexdigest()[:6]
            colors.append(hex_color)

        fig, ax = plt.subplots(figsize=(10, 8))

        values = [v for _, v in final_items]
        labels = [l for l, _ in final_items]

        wedges, _ = ax.pie(values, colors=colors, startangle=90, radius=1.2)

        for i, (wedge, label, value) in enumerate(zip(wedges, labels, values)):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = math.cos(math.radians(angle))
            y = math.sin(math.radians(angle))

            if label != "Other" and value >= internal_threshold:
                # Internal label
                ax.text(x * 0.7, y * 0.7, f"{label}\n({value:.1f}%)",
                        ha='center', va='center', fontsize=9)
            else:
                # External annotation
                ha = 'left' if x >= 0 else 'right'
                ax.annotate(
                    f"{label} ({value:.1f}%)",
                    xy=(x, y),
                    xytext=(1.4 * x, 1.4 * y),
                    ha=ha, va='center',
                    arrowprops=dict(arrowstyle='-', connectionstyle="arc3"),
                    fontsize=8
                )

        ax.set_title('Distribution of Most Similar Reference Sections', pad=20)
        ax.axis('equal')
        plt.tight_layout()

        plt.savefig("img/" + self.title + "_pie.png")
        return fig


######## Bar Chart Animations

    def animate_section_winners(self, interval=200):
        """
        Creates an animation showing how section winners accumulate over time.

        Args:
            save_path (str, optional): Path to save the animation as a GIF. If None, animation is displayed inline.
            interval (int, optional): Time interval between frames in milliseconds.

        Returns:
            matplotlib.animation.Animation: The created animation object
        """

        # Set up the initial figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Dictionary to track winner counts across frames
        win_counts = {}
        # List to store animation frames
        frames = []
        # Track winners in order of first appearance
        ordered_winners = []

        # Prepare data for animation
        for i, section in enumerate(self.sections):
            # Create similarity calculator for this section
            calc = SimilarityCalculator(section, self.reference_dict)
            # Get similarities and find the winner
            similarities = calc.compute_similarities(normalize=True)
            winner = max(similarities.items(), key=lambda x: x[1])[0]

            # Update win counts
            win_counts[winner] = win_counts.get(winner, 0) + 1

            # Add to ordered list of unique winners (if not already there)
            if winner not in ordered_winners:
                ordered_winners.append(winner)

            # Create a copy of the current win_counts for this frame
            frame_data = win_counts.copy()
            frames.append((frame_data, winner, ordered_winners.copy()))

        # Determine the maximum count for y-axis scaling
        max_count = max(win_counts.values())

        # Define animation update function
        def update(frame_idx):
            ax.clear()

            frame_data, current_winner, current_winners = frames[frame_idx]

            # Get data for this frame
            labels = current_winners
            values = [frame_data.get(label, 0) for label in labels]

            # Create bars with different colors based on which one was just incremented
            bars = ax.bar(labels, values, color=['lightblue' if label != current_winner else 'orange' for label in labels])

            # Add count labels on top of bars
            for bar, count in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom')

            # Add frame title and information
            section_num = frame_idx + 1
            ax.set_title(f'Section Winners (Section {section_num}/{len(self.sections)})\n'
                        f'Current winner: {current_winner}', fontsize=12)

            # Set labels and limits
            ax.set_xlabel('Reference Categories', fontsize=12)
            ax.set_ylabel('Number of Sections Won', fontsize=12)
            ax.set_ylim(0, max_count + 1)  # Add some padding

            # Force y-axis to use integers only
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')

            # Ensure layout is tight
            plt.tight_layout()

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=True)

        # Handle saving or displaying
        save_path = "img/" + self.title + "_winners.gif"
        ani.save(save_path, writer='pillow', fps=1000/interval)
        plt.close()

        return ani

    def animate_section_winners_bar(self, period_length=100):
        """
        Creates a bar chart race animation showing how section winners accumulate over time.

        Args:
            save_path (str, optional): Path to save the animation as a GIF. If None, animation is displayed inline.
            period_length (int, optional): Duration of each period (frame) in milliseconds.

        Returns:
            None
        """

        save_path = "img/" + self.title + "_winners_bcr.mp4"

        # Dictionary to track winner counts across frames
        win_counts = {}

        # List of dictionaries to collect cumulative data for each section
        cumulative_data = []

        # Set of all unique winners for consistent columns
        all_winners = set()

        for section in self.sections:
            # Create similarity calculator for this section
            calc = SimilarityCalculator(section, self.reference_dict)

            # Compute similarities and find the winner
            similarities = calc.compute_similarities(normalize=True)
            winner = max(similarities.items(), key=lambda x: x[1])[0]

            # Update win counts
            win_counts[winner] = win_counts.get(winner, 0) + 1

            # Track all unique winners
            all_winners.update(win_counts.keys())

            # Append a snapshot of the current counts
            snapshot = {winner: win_counts.get(winner, 0) for winner in all_winners}
            cumulative_data.append(snapshot)

        # Create DataFrame from cumulative data
        df = pd.DataFrame(cumulative_data).fillna(0).astype(int)

        # Add index as section number labels
        df.index = [f'Section {i+1}' for i in range(len(df))]

        # Generate the bar chart race
        bcr.bar_chart_race(
            df=df,
            filename=save_path,
            orientation='h',
            sort='desc',
            n_bars=10,
            fixed_order=False,
            fixed_max=True,
            steps_per_period=5,
            period_length=period_length,
            interpolate_period=False,
            bar_size=.95,
            period_label={'x': .99, 'y': .15, 'ha': 'right', 'va': 'center'},
            figsize=(12, 7),
            dpi=144,
            cmap='dark12',
            title='Cumulative Section Winners Over Time'#,
            #bar_label_fmt='{:.0f}',
        )

######## Nearest Reference

    def sort_sections_by_reference(self, reference_embedding):
      """
      Sort sections based on their similarity to a reference embedding.

      Args:
          reference_embedding (np.ndarray): The reference embedding to compare against

      Returns:
          list: A list of Section objects sorted by decreasing similarity to the reference embedding
      """
      sections_with_similarities = []

      for section in self.sections:
          # Calculate cosine similarity between section embedding and reference embedding
          # Cosine similarity = dot(u,v) / (||u||*||v||)
          dot_product = np.dot(section.embedding, reference_embedding)
          norm_section = np.linalg.norm(section.embedding)
          norm_reference = np.linalg.norm(reference_embedding)

          # Calculate cosine similarity (avoid division by zero)
          if norm_section > 0 and norm_reference > 0:
              similarity = dot_product / (norm_section * norm_reference)
          else:
              similarity = 0

          # Store similarity in section's metadata
          section.add_metadata('reference_similarity', float(similarity))

          # Add tuple of (section, similarity) to list
          sections_with_similarities.append((section, similarity))

      # Sort sections by similarity in descending order
      sections_with_similarities.sort(key=lambda x: x[1], reverse=True)

      # Extract just the sorted sections
      sorted_sections = [item[0] for item in sections_with_similarities]

      return sorted_sections

####### Averages

    def average_section_embedding(self):
      """
      Calculate the average embedding across all sections.

      Returns:
          np.ndarray: The average embedding vector
      """
      # Stack all embeddings into a single array
      all_embeddings = np.stack([section.embedding for section in self.sections])
      # Calculate the average along the first axis (across all sections)
      avg_embedding = np.mean(all_embeddings, axis=0)
      return avg_embedding

    def normalize_section_embeddings(self):
      """
      Normalize each section's embedding by dividing by the average embedding.
      Updates the section.embedding values in place.

      Returns:
          list: List of normalized embeddings
      """
      avg_embedding = self.average_section_embedding()
      normalized_embeddings = []

      for section in self.sections:
          # Avoid division by zero by adding a small epsilon
          normalized_embedding = section.embedding / (avg_embedding + 1e-10)
          section.embedding = normalized_embedding
          normalized_embeddings.append(normalized_embedding)

      # Update the reference dictionary
      self.reference_dict = {section.text: section.embedding for section in self.sections}
      return normalized_embeddings

    def section_similarities_to_average(self):
      """
      Calculate cosine similarity of each section's embedding to the average embedding.

      Returns:
          list: List of cosine similarity values
      """
      avg_embedding = self.average_section_embedding()
      similarities = []

      for section in self.sections:
          # Calculate cosine similarity between section embedding and average embedding
          # Cosine similarity = dot(u,v) / (||u||*||v||)
          dot_product = np.dot(section.embedding, avg_embedding)
          norm_section = np.linalg.norm(section.embedding)
          norm_avg = np.linalg.norm(avg_embedding)

          # Calculate cosine similarity (avoid division by zero)
          if norm_section > 0 and norm_avg > 0:
              similarity = dot_product / (norm_section * norm_avg)
          else:
              similarity = 0

          similarities.append(similarity)

          # Also store in the section's metadata
          section.add_metadata('avg_similarity', float(similarity))

      return similarities

########### Nearest Neighbors

    def get_nearest_neighbors(self, query_section, k=5, include_self=False):
        """
        Find the K nearest neighbors to a query section based on embedding cosine similarity.

        Args:
            query_section (Section): The query section to find neighbors for
            k (int): The number of neighbors to return
            include_self (bool): Whether to include the query section in the results
                                (if it exists in the collection)

        Returns:
            list: A list of tuples (section, similarity) sorted by decreasing similarity
        """
        sections_with_similarities = []
        query_embedding = query_section.embedding

        for section in self.sections:
            # Skip if it's the same section and we don't want to include self
            if not include_self and section.narrative_id == query_section.narrative_id and section.section_index == query_section.section_index:
                continue

            # Calculate cosine similarity between query embedding and section embedding
            dot_product = np.dot(query_embedding, section.embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_section = np.linalg.norm(section.embedding)

            # Calculate cosine similarity (avoid division by zero)
            if norm_query > 0 and norm_section > 0:
                similarity = dot_product / (norm_query * norm_section)
            else:
                similarity = 0

            # Add tuple of (section, similarity) to list
            sections_with_similarities.append((section, similarity))

        # Sort sections by similarity in descending order
        sections_with_similarities.sort(key=lambda x: x[1], reverse=True)

        # Return the top k sections with their similarities
        return sections_with_similarities[:k]
