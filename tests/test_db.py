import uuid

import pytest
from sqlalchemy import select, text

from app.db.models import Document, DocumentChunk, Tenant


@pytest.mark.asyncio
async def test_db_connection(db_session):
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_pgvector_extension_enabled(db_session):
    result = await db_session.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    assert result.scalar() == "vector", "pgvector extension is not enabled"


@pytest.mark.asyncio
async def test_can_create_tenant(db_session):
    tenant = Tenant(name="test-company", password_hash="placeholder_hash")
    db_session.add(tenant)
    await db_session.flush()  # write to DB but don't commit (rollback will undo this)

    result = await db_session.execute(
        select(Tenant).where(Tenant.name == "test-company")
    )
    fetched = result.scalar_one()
    assert fetched.id is not None
    assert fetched.is_active is True


@pytest.mark.asyncio
async def test_can_create_document_chunk_with_embedding(db_session):
    tenant = Tenant(name="embedding-test-tenant", password_hash="placeholder_hash")
    db_session.add(tenant)
    await db_session.flush()

    document = Document(
        tenant_id=tenant.id,
        filename="test.pdf",
        content_type="application/pdf",
        content_hash="a" * 64,
    )
    db_session.add(document)
    await db_session.flush()

    fake_embedding = [0.1] * 384

    chunk = DocumentChunk(
        tenant_id=tenant.id,
        document_id=document.id,
        chunk_index=0,
        chunk_text="This is a test chunk of text.",
        embedding=fake_embedding,
    )
    db_session.add(chunk)
    await db_session.flush()

    result = await db_session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == document.id)
    )
    fetched_chunk = result.scalar_one()
    assert fetched_chunk.embedding is not None
    assert len(fetched_chunk.embedding) == 384


@pytest.mark.asyncio
async def test_tenant_isolation_enforced(db_session):
    """Verifies that querying by tenant_id returns ONLY that tenant's data."""
    tenant_a = Tenant(name="tenant-alpha", password_hash="placeholder_hash")
    tenant_b = Tenant(name="tenant-beta", password_hash="placeholder_hash")
    db_session.add_all([tenant_a, tenant_b])
    await db_session.flush()

    doc_a = Document(tenant_id=tenant_a.id, filename="alpha.pdf", content_type="application/pdf", content_hash="a" * 64)
    doc_b = Document(tenant_id=tenant_b.id, filename="beta.pdf", content_type="application/pdf", content_hash="b" * 64)
    db_session.add_all([doc_a, doc_b])
    await db_session.flush()

    result_a = await db_session.execute(
        select(Document).where(Document.tenant_id == tenant_a.id)
    )
    docs_a = result_a.scalars().all()
    assert len(docs_a) == 1
    assert docs_a[0].filename == "alpha.pdf"

    result_b = await db_session.execute(
        select(Document).where(Document.tenant_id == tenant_b.id)
    )
    docs_b = result_b.scalars().all()
    assert len(docs_b) == 1
    assert docs_b[0].filename == "beta.pdf"

    result_all = await db_session.execute(select(Document))
    all_docs = result_all.scalars().all()
    assert len(all_docs) == 2  # both exist in DB, but tenants must never see each other's


@pytest.mark.asyncio
async def test_cascade_delete_removes_chunks(db_session):
    tenant = Tenant(name="cascade-test-tenant", password_hash="placeholder_hash")
    db_session.add(tenant)
    await db_session.flush()

    document = Document(
        tenant_id=tenant.id, filename="deleteme.pdf", content_type="application/pdf", content_hash="c" * 64
    )
    db_session.add(document)
    await db_session.flush()

    chunk = DocumentChunk(
        tenant_id=tenant.id,
        document_id=document.id,
        chunk_index=0,
        chunk_text="This chunk should be deleted with its document.",
    )
    db_session.add(chunk)
    await db_session.flush()

    doc_id = document.id

    await db_session.delete(document)
    await db_session.flush()

    result = await db_session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    )
    assert result.scalars().all() == []
