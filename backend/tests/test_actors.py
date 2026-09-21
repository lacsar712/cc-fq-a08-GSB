"""Unit tests for Actor pipeline (no DB required)."""

import asyncio

import pytest

from app.pipeline.actors import (
    ActorError,
    FastqRead,
    GcContentActor,
    NContentActor,
    ParseActor,
    PipelineContext,
    QualityHistActor,
    QueueMessage,
    ReportActor,
)
from app.pipeline.runner import _run_chain


GOOD_FASTQ = """@SEQ1
ACGTACGT
+
IIIIHHHH
@SEQ2
NNNNACGT
+
IIIIIIII
"""

BROKEN_FASTQ = """@SEQ1
ACGT
NOTPLUS
IIII
"""


@pytest.mark.asyncio
async def test_parse_actor_rejects_malformed():
    actor = ParseActor()
    in_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    await in_q.put(QueueMessage(ok=True, context=PipelineContext(fastq_text=BROKEN_FASTQ)))
    await actor.run(in_q, out_q)
    result = await out_q.get()
    assert result.ok is False
    assert "必须以 +" in (result.error or "")


@pytest.mark.asyncio
async def test_parse_actor_ok_and_quality_mean():
    ok, ctx, stages = await _run_chain(GOOD_FASTQ)
    assert ok is True
    assert stages["ParseActor"]["status"] == "success"
    assert stages["GcContentActor"]["status"] == "success"
    assert stages["ReportActor"]["status"] == "success"
    assert ctx.metrics["reads"] == 2
    assert "mean_quality" in ctx.metrics
    assert ctx.metrics["mean_quality"] > 0
    assert ctx.metrics["n_rate"] == 0.25  # 4 N out of 16 bases
    # SEQ1: 4 GC/8 bases; SEQ2: 2 GC over 4 canonical bases (4 N excluded)
    assert ctx.metrics["gc_count"] == 6
    assert ctx.metrics["gc_at_bases"] == 12
    assert ctx.metrics["gc_rate"] == 0.5
    assert ctx.metrics["report"]["gc_rate"] == 0.5
    assert ctx.metrics["summary"]["gc_rate"] == 0.5


@pytest.mark.asyncio
async def test_gc_actor_counts_only_canonical_bases():
    actor = GcContentActor()
    in_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    ctx = PipelineContext(
        fastq_text="",
        reads=[
            FastqRead(header="@x", sequence="ggccnnACGT", plus="+", quality="I" * 10),
        ],
    )
    await in_q.put(QueueMessage(ok=True, context=ctx))
    await actor.run(in_q, out_q)
    result = await out_q.get()
    assert result.ok is True
    assert result.context.metrics["gc_count"] == 6  # g,g,c,c + C,G = 6
    assert result.context.metrics["gc_at_bases"] == 8  # 10 bases minus 2 N
    assert result.context.metrics["gc_rate"] == 0.75


@pytest.mark.asyncio
async def test_broken_stops_pipeline():
    ok, ctx, stages = await _run_chain(BROKEN_FASTQ)
    assert ok is False
    assert stages["ParseActor"]["status"] == "failed"
    assert stages["QualityHistActor"]["status"] == "skipped"
    assert stages["NContentActor"]["status"] == "skipped"
    assert stages["GcContentActor"]["status"] == "skipped"
    assert stages["ReportActor"]["status"] == "skipped"
    assert ctx.failed_actor == "ParseActor"


def test_parse_length_mismatch():
    actor = ParseActor()
    with pytest.raises(ActorError, match="长度不一致"):
        actor._parse("@A\nACGT\n+\nII\n")
