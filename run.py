import os
import glob
import json
import random
import numpy as np
from sklearn.decomposition import PCA
#
from analysis.tsne_static import plot_section_embeddings
#
from narrative_structures.narrative import Narrative
from narrative_structures.section_collection import SectionCollection
#
from embeddings.embedding_cache import EmbeddingCache
from embeddings.similarity_calculator import SimilarityCalculator
#

######## References

def get_cached_references():
    #
    cache = EmbeddingCache(cache_dir="embeddings/data/cache_screenplay")
    #
    # make embeddings via API
    #with open('embeddings/data/reference_values_lit.json', 'r') as f:
    #    reference_values = json.load(f)
    #reference_dict = cache.get_reference_dictionary(reference_values)
    #
    # save in cache
    #cache.save_reference_dict(reference_dict, "narrative_elements")
    #
    # load from cache
    references = cache.load_reference_dict("narrative_elements")
    return cache.embedder.order_embeddings_by_similarity(references) # order by embedding similarity, so that on e.g. heatmap, more-similar references will be together and hence more visibly legible
    #return references

########## Individual Narrative Analysis #############

def heatmap(narrative, references, title):
    calculation_results = []
    for section in narrative.sections:
        calculator = SimilarityCalculator(section, references)
        similarities = calculator.compute_similarities()
        calculation_results.append(similarities)
    calculator.plot_similarity_heatmap(calculation_results, path_title=title)

def radio(section, references, title):
    calculator = SimilarityCalculator(section, references)
    similarities = calculator.compute_similarities()
    calculator.plot_radio(similarities, save_path="img/" + title + "_radio.png")

def analyze_individual_narrative(title="thor", filepath="texts/marvel/thor-2011.pdf", num_sections=50, perplexity=5): # perplexity will be used for "animated_combined" tSNE
    # get references
    references = get_cached_references()
    # get narrative
    narrative = Narrative(filepath, num_sections=num_sections, perplexity=perplexity, title=title)
    # radio plot example
    radio(narrative.sections[-1], references, title)
    # order sections by nearest to reference
    ranked_sections = narrative.sort_sections_by_reference(references["Intense argument"])
    print("Nearest Section to Reference: ")
    print(ranked_sections[0].text)
    # animate narrative
    narrative.animate_combined()
    # heatmap
    # uncomment either call - pruned shows winners only, unpruned is "true" value
    heatmap(narrative, references, title) # not pruned
    #heatmap(narrative, narrative.reference_dict, title) # pruned
    # override reference dict with full set of references
    # by default, reference dict is section_text:embedding, used for tSNE animation
    narrative.reference_dict = references
    # make pie chart
    win_counts = narrative.get_section_winners(prune_references=True)
    narrative.create_winners_pie_chart(win_counts)
    # make bar race
    narrative.animate_section_winners_bar()

#get_cached_references()
analyze_individual_narrative()

######### Multiple Narrative Analysis ##########

def get_all_narratives(pdf_dir="texts/marvel", num_sections=10, truncate=0):
    # files
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if truncate > 0:
        pdf_files = pdf_files[0:truncate]
    # load
    narratives = []
    for path in pdf_files:
        narrative = Narrative(path, num_sections=num_sections, perplexity=5)
        #narrative.normalize_section_embeddings()
        narratives.append(narrative)
    return narratives

def assign_embeddings_to_sections(narratives):
    references = get_cached_references()
    for narrative in narratives:
        for section in narrative.sections:
            calculator = SimilarityCalculator(section, references)
            concat_embedding = calculator.compute_similarity_weighted_embedding(method="top_k_weighted", top_k=2)
            section.embedding = concat_embedding

def analyze_collection_narratives(title="marvel", pdf_dir="texts/marvel", num_sections=50, truncate=0, perplexity=50):

    # load
    narratives = get_all_narratives(pdf_dir=pdf_dir, num_sections=num_sections, truncate=truncate)
    assign_embeddings_to_sections(narratives)

    # plot tsne sections
    sections = sum((n.sections for n in narratives), [])
    plot_section_embeddings(sections, perplexity=perplexity, path_title=title)

    # all narratives: charts
    collection = SectionCollection(title, sections)
    collection.reference_dict = get_cached_references()

    # pie
    win_counts = collection.get_section_winners(prune_references=True)
    collection.create_winners_pie_chart(win_counts)

    # bar
    collection.animate_section_winners_bar()

#analyze_collection_narratives()
