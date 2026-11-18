"""The cost event must be gated on WINNING the terminal write.

The money bug this pins: ``_execute`` called ``mark_completed`` and then
``_emit_cost`` unconditionally. ``mark_completed`` was an unguarded
``UPDATE ... SET status='completed'``, so this sequence billed twice for one analysis:

    job strands past STRANDED_AFTER_SECONDS inside the vision stage (which writes no
    progress between frames) -> a client poll reaps it to `failed` -> the client
    resubmits, and pays for a SECOND analysis -> the still-live original finally
    finishes, flips the row back to `completed`, and emits a SECOND cost event for the
    same job_id.

``video_compiler_service`` already does this correctly — ``api/routes/compile.py`` does
``won = await _stamp_completed(...)`` then ``if won: _emit_cost(...)``. This mirrors it.

Both directions are asserted deliberately. A test that only checks the lost-write case
would still pass if the boolean were inverted — and an inverted gate silently stops ALL
billing, which is a worse failure than the one being fixed.
"""

from __future__ import annotations

import inspect

from src.services.pipeline import PipelineRunner


def test_execute_gates_emit_cost_on_the_terminal_write() -> None:
    """The CALL SITE is what has to be asserted, so this reads ``_execute``'s source.

    Honest about what this is and why. The gate is a control-flow branch near the end of
    a long pipeline whose happy path needs ffmpeg, a vision provider, transcription and
    an object store — reproducing it end-to-end would test the fakes, not the branch. And
    a test that drives a helper with a hand-written ``if won:`` proves only that the test
    author can write an if-statement: it is exactly the class of vacuous test that
    video_compiler_service #153 called out, where the load-bearing assertion had to be on
    the call site rather than on the helper.

    So this pins the four things that must remain true:
      1. ``_execute`` still routes its completion through ``_complete_and_bill``,
      2. ``mark_completed``'s return value is captured there (it used to return None),
      3. there is an ``if not won:`` early-return,
      4. it comes BEFORE ``_emit_cost``.
    Break any one and the double-bill is back.
    """
    execute_src = inspect.getsource(PipelineRunner._execute)
    assert "self._complete_and_bill(" in execute_src, (
        "_execute no longer routes completion through the gated helper"
    )

    source = inspect.getsource(PipelineRunner._complete_and_bill)
    assert "won = await JobRepository(session).mark_completed(" in source, (
        "mark_completed's won-boolean is no longer captured"
    )
    assert "if not won:" in source, "the won-boolean is captured but never checked"
    assert source.index("if not won:") < source.index("self._emit_cost("), (
        "_emit_cost is no longer guarded by the won-boolean — this is the double-bill"
    )
