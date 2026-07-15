"""Channel semantics: the reduction rules that make the BSP barrier well-defined."""

from __future__ import annotations

from operator import add
from typing import Annotated

import pytest

from tensorsketch import BinaryOperatorAggregate, LastValue, Reducer, Schema, Topic
from tensorsketch.core.channels import channel_for_field
from tensorsketch.core.errors import EmptyChannelError, InvalidUpdateError


def test_lastvalue_reads_after_write() -> None:
    c: LastValue[int] = LastValue()
    assert not c.is_set
    with pytest.raises(EmptyChannelError):
        c.get()
    assert c.update([5]) is True
    assert c.get() == 5
    assert c.is_set


def test_lastvalue_rejects_two_writes_in_one_step() -> None:
    c: LastValue[int] = LastValue()
    with pytest.raises(InvalidUpdateError):
        c.update([1, 2])


def test_lastvalue_empty_update_is_noop() -> None:
    c: LastValue[int] = LastValue()
    assert c.update([]) is False
    assert not c.is_set


def test_aggregate_seeds_then_folds() -> None:
    c: BinaryOperatorAggregate[int] = BinaryOperatorAggregate(add)
    assert c.update([1]) is True
    assert c.get() == 1
    c.update([2, 3])  # folds within a step, and across steps
    assert c.get() == 6


def test_topic_concatenates_and_accumulates() -> None:
    c: Topic[str] = Topic()
    assert c.get() == []  # always set; empty by default
    c.update([["a", "b"]])  # each update is a batch (list) to append
    c.update([["c"]])
    assert c.get() == ["a", "b", "c"]


def test_topic_without_accumulate_holds_last_step_only() -> None:
    c: Topic[str] = Topic(accumulate=False)
    c.update([["a"]])
    assert c.get() == ["a"]
    c.update([["b"]])
    assert c.get() == ["b"]


def test_channel_for_field_picks_channel_by_annotation() -> None:
    class S(Schema):
        plain: int
        summed: Annotated[list[int], Reducer(add)]
        stream: Annotated[list[int], Topic()]

    fields = S.model_fields
    assert isinstance(channel_for_field(fields["plain"]), LastValue)
    assert isinstance(channel_for_field(fields["summed"]), BinaryOperatorAggregate)
    assert isinstance(channel_for_field(fields["stream"]), Topic)


def test_channel_for_field_returns_fresh_topic_each_call() -> None:
    class S(Schema):
        stream: Annotated[list[int], Topic()]

    field = S.model_fields["stream"]
    first = channel_for_field(field)
    first.update([[1]])
    second = channel_for_field(field)
    assert second.get() == []  # no state leaked from the first channel
