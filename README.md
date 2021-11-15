# EAT
Embedding-based annotation transfer (EAT) uses Euclidean distance between vector representations (embeddings) of proteins to transfer annotations from a set of labeled lookup protein embeddings to query protein embeddings.


# Abstract 
Here, we present a novel approach that expands the concept of homology-based inference (HBI) from a low-dimensional sequence-distance lookup to the level of a high-dimensional embedding-based annotation transfer (EAT). More specifically, we replace sequence similarity as means to transfer annotations from one set of proteins (lookup; usually labeled) to another set of proteins (queries; usually unlabeled) by Euclidean distance between single protein sequence representations (embeddings) from protein Language Models (pLMs). Secondly, we introduce a novel set of embeddings (dubbed ProtTucker) that were optimized towards constraints captured by hierarchical classifications of protein 3D structures (CATH). These new embeddings enabled the intrusion into the midnight zone of protein comparisons, i.e., the region in which the level of pairwise sequence similarity is akin of random relations and therefore is hard to navigate by HBI methods. Cautious benchmarking showed that ProtTucker reached further than advanced sequence comparisons without the need to compute alignments allowing it to be orders of magnitude faster.

# Getting started
Install bio_embeddings package as described here: https://github.com/sacdallago/bio_embeddings


Clone the EAT repository and get started as described in the Usage section below:
   ```sh
   git clone https://github.com/Rostlab/EAT.git
   ```

# Usage
- Quick start: General purpose (1 nearest neighbor/1-NN) without additional labels files:

For general annotation transfer/nearest-neighbor search in embedding space, the pLM ProtT5 is used. It was only optimized using raw protein sequences (self-supervised pre-training) and is therefor not biased towards a certain task. The following command will take two FASTA files holding protein sequences as input (lookup & queries) in order to transfer annotations (fasta headers) from lookup to queries:

```sh
python eat.py --lookup data/example_data_subcell/deeploc_lookup.fasta --queries data/example_data/la_query_setHARD.fasta --output eat_results/
```
- Extended: General purpose (3-NN) with additional labels:

If you want to provide your labels as separate file (labels are expected to have CSV format with 1st col being the fasta header and 2nd col being the label) and retrieve the first 3 nearest-neighbors (NN) instead of only the single NN:


```sh
python eat.py --lookup data/example_data_subcell/deeploc_lookup.fasta --queries data/example_data/la_query_setHARD.fasta --output eat_results/ --lookupLabels data/example_data_subcell/deeploc_lookup_labels.txt --queryLabels data/example_data_subcell/la_query_setHARD_labels.txt
```
Example output is given here: https://github.com/Rostlab/EAT/blob/main/data/example_data_subcell/example_output_protT5_NN3.txt

- Expert solution tailored for remote homology detection:
For remote homology detection, we recommend to use ProtTucker(ProtT5) embeddings that were specialized on capturing the CATH hierarchy:

```sh
python eat.py --lookup data/example_data_subcell/deeploc_lookup.fasta --queries data/example_data/la_query_setHARD.fasta --output eat_results/ --use_tucker 1
```


# Figures
<img src="https://github.com/Rostlab/EAT/blob/main/ProtTucker_tSNE.png?raw=true" width="60%" height="60%">
Contrastive learning improved CATH class-level clustering. Using t-SNE, we projected the high-dimensional embedding space onto 2D before (left; ProtT5) and after (right, ProtTucker(ProtT5)) contrastive learning. The colors mark the major class level of CATH (C) distinguishing proteins according to their major distinction in secondary structure content.

<br/><br/>


<img src="https://github.com/Rostlab/EAT/blob/main/ProtTucker_reliability.png?raw=true" width="50%" height="50%">
Similar to varying E-value cut-offs for homology-based inference (HBI), we examined whether the fraction of correct predictions (accuracy; left axis) depended on embedding distance (x-axis) for EAT. Toward this end, we transferred annotations for all four levels of CATH (Class: blue; Architecture: orange; Topology: green; Homologous superfamily: red) from proteins in our lookup set to the queries in our test set using the hit with smallest Euclidean distance. The fraction of test proteins having a hit below a certain distance threshold (coverage, right axis, dashed lines) was evaluated separately for each CATH level. For example, at a Euclidean distance of 1.1 (marked by black vertical dots), 78% of the test proteins found a hit at the H-level (Cov(H)=78%) and of 89% were correctly predicted (Acc(H)=89%). Similar to decreasing E-values for HBI, decreasing embedding distance correlated with EAT performance. This correlation importantly enables users to select only the, e.g., 10% top hits, or all hits with an accuracy above a certain threshold.


# Reference
tbd
