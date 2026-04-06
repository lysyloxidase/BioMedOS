# BioMedOS

> Local-first multi-agent biomedical operating system with knowledge graph reasoning, PubMed RAG, graph ML, and clinical decision support. Your private biomedical AI assistant.

BioMedOS combines a biomedical knowledge graph, literature retrieval, graph machine learning, clinical tooling, and a 12-agent orchestration layer into one local stack. It is designed to run with Ollama, ChromaDB, and Python on your own machine without cloud model dependencies.

## Why BioMedOS?

- Local-first by design: Ollama, ChromaDB, and the core graph stack run on local infrastructure with no required cloud LLM APIs.
- KG + RAG hybrid reasoning: retrieval merges BM25, dense search, and knowledge graph context so answers are grounded in both literature and structure.
- 12 specialized agents: routing, literature synthesis, graph traversal, link prediction, repurposing, clinical reasoning, review writing, and verification are all separated into focused modules.
- Clinical decision support built in: DDI checks, phenotype mapping, differential diagnosis, contraindication review, and evidence grading are part of the same platform.

## Architecture

```text
                         +----------------------------------+
                         |           Web UI / API           |
                         |  SPA + FastAPI + WebSocket chat  |
                         +----------------+-----------------+
                                          |
                           +--------------v--------------+
                           |       Agent Orchestration    |
                           | Router -> Specialists ->     |
                           | Sentinel -> Aggregation      |
                           +--------------+--------------+
                                          |
        +------------------------+--------+---------+------------------------+
        |                        |                  |                        |
+-------v--------+     +---------v---------+ +------v------+     +-----------v-----------+
|  Data Sources  |     | Knowledge Graph   | | RAG Engine  |     | Clinical Tooling      |
| PubMed, OT,    | --> | NetworkX schema   | | BM25+dense  | --> | DDI, phenotypes,      |
| ChEMBL, HPO... |     | queries, stats,   | | KG context, |     | diagnosis, evidence   |
|                |     | graph ML export   | | reranking   |     | grading               |
+----------------+     +---------+---------+ +------+------+     +-----------+-----------+
                                          |                  |
                                 +--------v------------------v--------+
                                 |     Graph ML + Local Ollama        |
                                 | GraphSAGE, R-GCN, Node2Vec, LLMs   |
                                 +------------------------------------+
```

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -e .[dev]
python scripts/demo.py
python scripts/run_local_api.py --port 8010
```

Alternative entrypoints:

```bash
docker compose up -d
python scripts/build_graph.py --genes EGFR TP53 BRCA1 ALK --sources open_targets string_db pubmed
python scripts/train_models.py --model graphsage --epochs 50 --edge-type gene_associated_with_disease
uvicorn biomedos.api.app:app --reload
```

## Data Sources

| API | Purpose |
| --- | --- |
| PubMed E-utilities | Search and fetch abstracts for literature grounding |
| PubTator | Biomedical annotations for literature enrichment |
| Open Targets | Gene-disease associations, tractability, and drug evidence |
| ChEMBL | Drug metadata, mechanisms, bioactivities, and ADMET signals |
| UniProt | Protein metadata and gene-to-protein lookup |
| STRING DB | Protein-protein interaction context |
| RxNorm | Drug normalization, interactions, and NDC mappings |
| Human Phenotype Ontology | Phenotype search, disease links, and phenotype similarity |
| OpenFDA | Adverse events, labels, contraindications, and recalls |
| DisGeNET | Gene-disease association evidence |
| ClinicalTrials.gov v2 | Trial metadata for translational and clinical workflows |

## Knowledge Graph Schema

Node types:
`Gene`, `Protein`, `Disease`, `Drug`, `Compound`, `Pathway`, `Phenotype`, `CellType`, `Tissue`, `SideEffect`, `ClinicalTrial`, `Publication`

Edge types:
`gene_associated_with_disease`, `gene_in_pathway`, `gene_interacts_with_gene`, `protein_interacts_with_protein`, `drug_targets_gene`, `drug_treats_disease`, `drug_interacts_with_drug`, `drug_causes_side_effect`, `compound_binds_target`, `disease_has_phenotype`, `disease_involves_pathway`, `gene_expressed_in_tissue`, `pathway_crosstalks_with`, `publication_mentions_gene`, `publication_mentions_disease`, `drug_contraindicated_for`, `gene_associated_with_phenotype`, `trial_investigates_drug`

## Agents

| Agent | Default model | Role |
| --- | --- | --- |
| Router | `llama3.2:3b` | Classifies requests and decomposes complex workflows |
| Literature | `qwen2.5:14b` | Runs PubMed-grounded retrieval and cited synthesis |
| Graph Explorer | `qwen2.5:14b` | Traverses paths, neighborhoods, and local subgraphs |
| Link Predictor | local GNN + `qwen2.5:14b` | Scores novel graph links and explains predictions |
| Drug Repurposer | local GNN + `qwen2.5:14b` | Finds cross-disease drug opportunities with confidence labels |
| Geneticist | `qwen2.5:14b` | Builds gene-centric reports with network and druggability context |
| Pharmacologist | `qwen2.5:14b` | Reviews DDI, PK, ADMET, safety, and contraindications |
| Clinician | `qwen2.5:14b` | Produces differential diagnoses from symptoms and phenotypes |
| Pathway Analyst | `qwen2.5:14b` | Runs enrichment and pathway crosstalk analysis |
| Hypothesis Generator | `qwen2.5:14b` | Surfaces structural holes and ranked mechanistic hypotheses |
| Review Writer | `qwen2.5:14b` | Drafts narrative reviews with citations and self-critique |
| Sentinel | `phi4:14b` | Verifies claims, citations, and hallucination risk |

## Clinical Decision Support

- Drug-drug interaction checking via RxNorm and OpenFDA
- Symptom to HPO term mapping with phenotype matching
- Differential diagnosis combining phenotype evidence, KG support, and literature
- Contraindication review from local label parsing
- Evidence grading with GRADE-style classifications

## Graph ML

| Model | Purpose | Status | Metrics |
| --- | --- | --- | --- |
| GraphSAGE | Heterogeneous link prediction baseline | Implemented | See [RESULTS.md](RESULTS.md) |
| R-GCN | Multi-relational reasoning with relation-specific weights | Implemented | Placeholder |
| Node2Vec | CPU-friendly embedding baseline | Implemented | Placeholder |

## RAG Pipeline

```text
Query
  -> BM25 sparse retrieval
  -> Dense vector retrieval
  -> Knowledge graph context extraction
  -> Reciprocal rank fusion
  -> Cross-encoder reranking
  -> Grounded answer generation with citations
```

## Comparison

| System | Local-first | Knowledge graph reasoning | Multi-agent orchestration | Clinical tooling |
| --- | --- | --- | --- | --- |
| BioMedOS | Yes | Yes | Yes | Yes |
| KG-RAG | Partial | Yes | No | No |
| PaperQA2 | Partial | No | Limited | No |
| STELLA | Partial | Limited | No | No |
| DrugAgent | Partial | Drug-focused | Yes | Limited |
| MedRAG | Partial | Limited | No | Limited |

## Hardware

- 32 GB RAM: practical baseline for 8B-class models and CPU-first experimentation
- 64 GB RAM: recommended for the default 14B reasoning and verification stack
- 128 GB RAM: suitable for larger local models, heavier indexing, and broader workflows

## Results

Benchmark placeholders and experiment templates live in [RESULTS.md](RESULTS.md).

## References

1. PubMed E-utilities documentation
2. Open Targets Platform
3. ChEMBL Web Services
4. STRING database
5. Human Phenotype Ontology
6. ClinicalTrials.gov API v2
7. PyTorch Geometric documentation
8. LangGraph documentation

## License

MIT. See [LICENSE](LICENSE).
