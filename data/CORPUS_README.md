# Enterprise Architecture RAG Corpus

This is the synthetic knowledge corpus for the Week 2 RAG project.

## Corpus contents
1. `api-design-standards.pdf`
2. `integration-architecture-guidelines.pdf`
3. `cloud-security-policy.pdf`
4. `logging-monitoring-standard.pdf`
5. `data-classification-policy.docx`
6. `architecture-principles.docx`
7. `incident-management.txt`
8. `api-error-codes.txt`

## Why mixed formats?
The corpus intentionally mixes PDF, DOCX, and TXT so the ingestion layer must use different loaders and normalize the content into a common document representation before chunking and embedding.

## Design intent
The documents include direct factual answers, related rules spread across multiple documents, semantic overlap between architecture principles and detailed standards, and exact identifiers such as `E2001` for later retrieval experiments.

Questions outside these documents should trigger the RAG application's fallback behavior rather than an invented answer.

All documents are synthetic training material for a fictional enterprise (Asteria Digital Services).
