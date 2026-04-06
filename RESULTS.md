# Results

This file is a placeholder for benchmark, ablation, and system validation results.

## End-to-End Benchmarks

| Workflow | Dataset / Scope | Primary metric | Result | Notes |
| --- | --- | --- | --- | --- |
| Literature QA | PubMed local index | Citation validity | TBD | Measure citation coverage and unsupported claims |
| Graph exploration | Demo KG / real KG | Path accuracy | TBD | Validate shortest paths and subgraph relevance |
| Link prediction | Held-out KG edges | AUROC / AUPRC / MRR | TBD | Report GraphSAGE, R-GCN, Node2Vec |
| Drug repurposing | Disease-specific case sets | Top-k precision | TBD | Exclude already-approved indications |
| Clinical reasoning | Symptom-to-diagnosis tasks | Top-1 / Top-5 accuracy | TBD | Combine HPO, KG, and literature evidence |
| Review writing | Topic-specific review prompts | Citation coverage / factuality | TBD | Sentinel audit required |

## Graph ML Model Comparison

| Model | Edge type | AUROC | AUPRC | Hits@10 | Hits@50 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GraphSAGE | `gene_associated_with_disease` | TBD | TBD | TBD | TBD | TBD |
| R-GCN | `gene_associated_with_disease` | TBD | TBD | TBD | TBD | TBD |
| Node2Vec | `gene_associated_with_disease` | TBD | TBD | TBD | TBD | TBD |

## RAG Ablations

| Configuration | BM25 | Dense | KG context | Reranker | Factuality | Citation validity |
| --- | --- | --- | --- | --- | --- | --- |
| Sparse only | Yes | No | No | No | TBD | TBD |
| Dense only | No | Yes | No | No | TBD | TBD |
| Hybrid | Yes | Yes | No | No | TBD | TBD |
| Hybrid + KG | Yes | Yes | Yes | No | TBD | TBD |
| Full pipeline | Yes | Yes | Yes | Yes | TBD | TBD |

## Clinical Validation

| Tool | Validation set | Metric | Result | Notes |
| --- | --- | --- | --- | --- |
| DDI checker | Curated drug pairs | Severity agreement | TBD | Compare RxNorm/OpenFDA synthesis |
| Phenotype matcher | Symptom vignettes | HPO mapping precision | TBD | Measure top-3 phenotype recovery |
| Differential diagnosis | Case vignettes | Recall@5 | TBD | Include KG and literature support |
| Contraindication checker | Drug labels | Label extraction accuracy | TBD | Compare against structured warnings |

## System Performance

| Task | Hardware tier | Latency | Peak RAM | Notes |
| --- | --- | --- | --- | --- |
| Router classification | 32 GB | TBD | TBD | Local 3B model |
| Literature synthesis | 64 GB | TBD | TBD | Hybrid retrieval plus generation |
| Review writing | 64 GB | TBD | TBD | Multi-section iterative drafting |
| GraphSAGE training | CPU | TBD | TBD | Demo graph, 50 epochs |
| Full workflow | 64 GB | TBD | TBD | Router -> specialists -> sentinel |
