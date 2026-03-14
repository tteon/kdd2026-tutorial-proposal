# Abstract Draft

Large language model systems increasingly rely on graph-structured knowledge to improve retrieval,
reasoning, and answer generation. However, the practical success of graph-grounded LLM systems depends
on more than retrieval alone. In domain-specific settings, downstream generation quality is strongly
affected by a sequence of upstream design choices: ontology selection, schema grounding, entity and
relation extraction, entity linking, graph materialization, and graph quality diagnostics. These stages
are often studied separately, making it difficult for researchers and practitioners to understand how
errors propagate from graph construction to final answers.

This tutorial presents a unified view of ontology-grounded graph construction and quality-aware
generation for domain-specific LLM systems. We connect research threads from knowledge graphs,
information extraction, GraphRAG, graph memory, and evaluation, and show how they fit into a single
end-to-end pipeline. A central theme of the tutorial is that ontology quality, extraction quality,
graph quality, and answer quality should be measured separately but interpreted jointly. We discuss
how profile-based ontology selection can be used to constrain extraction, how graph quality diagnostics
such as schema violations, duplicate entities, disconnected components, and missing query-support paths
can be attached as metadata, and how this metadata can guide query routing and evidence selection at
answer time.

We use finance as a running case study, with emphasis on governance, financial reporting, and
shareholder return, and discuss how FIBO-style profiles can support domain-aware graph construction.
The tutorial is intended for KDD researchers and practitioners interested in LLM systems, knowledge
graphs, and high-stakes domain applications, and aims to bridge current research ideas with practical
pipeline design and evaluation methodology.
