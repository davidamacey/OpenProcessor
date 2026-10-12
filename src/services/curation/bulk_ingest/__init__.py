"""Bulk ingest: one runner behind both the ``/ingest/bulk`` routes and the
``scripts/curation/bulk_ingest.py`` CLI (which is a client of those routes).

The runner drives the existing ingest pipeline through its one stable seam,
:meth:`~src.services.curation.ingest.CurationIngestService.ingest_batch`, so
every data-path improvement (inputs, transport, batching, GPUs) reaches bulk
ingest without a change here. What this package adds is orchestration: a
folder or manifest source, bounded concurrent batches, a durable ledger so a
run resumes, cancel, and progress with rates.
"""
