# State & channels

A TensorSketch graph has one piece of **state**, described by a `Schema`. But that state is not a plain
dictionary — under the hood, **each field is a channel**. A channel owns one value and knows
how to *fold in* new writes via a **reducer**. This is the mechanism that makes parallel
execution well-defined: when several nodes write in the same step, the reducer decides how
their writes combine.

## Declaring state

```python
from operator import add
from typing import Annotated
from tensorsketch import Schema, Reducer, Topic

class State(Schema):
    answer: str                                    # LastValue (the default)
    scratch: Annotated[list[str], Reducer(add)]    # accumulate concurrent + successive writes
    events: Annotated[list[str], Topic()]          # pub/sub append
    count: int = 0                                  # default → the channel starts at 0
```

The field's annotation picks its channel. No annotation → `LastValue`. A `Reducer(op)` or
`Topic()` marker in `Annotated[...]` selects a reducing channel.

## The channel types

### `LastValue` — keep the most recent write (default)

Holds a single value; reading before any write raises `EmptyChannelError`. **It rejects two
writes in the same superstep**, because the result would depend on scheduling order:

```python
# If NodeA and NodeB both write `answer` in the same step:
# InvalidUpdateError: LastValue channel received 2 writes in one superstep;
#                     give this field a reducer to combine concurrent writes
```

That error is a feature — it turns a hidden race into a loud, early failure. If concurrent
writes are intended, use a reducer.

### `BinaryOperatorAggregate` — fold with an operator

Selected by `Annotated[T, Reducer(op)]`. The first write seeds the value; every later write
(this step or a future one) is folded in as `value = op(value, update)`:

```python
total: Annotated[int, Reducer(add)]          # sums every write, across the whole run
scratch: Annotated[list[str], Reducer(add)]  # list + list = concatenation → accumulate
```

This is how a fan-in join collects results from parallel branches, and how a loop accumulates
a log across iterations.

### `Topic` — a stream of items

Selected by `Annotated[list[T], Topic()]`. Each write is a list that gets **concatenated** into
one growing list (so a node whose `Out` field is `list[str]` writes a small list, and it's
appended). Reads return the whole list. With `Topic(accumulate=False)` it holds only the current
step's writes (useful for one-shot fan-out payloads). A `Topic` is always "set" — it reads as
`[]` before any write.

## How writes become updates

A node returns an `Out` instance; each `Out` field is written to the state channel of the same
name. Within a superstep, writes are collected but **not visible** — they are applied together
at the **barrier** at the end of the step, each through its channel's reducer. So every node in
a step reads a consistent snapshot and can't see a sibling's mid-step write. (See the
[execution model](execution-model.md).)

## Seeding and reading state

- **Seeding:** `invoke(input)` writes your input into the matching channels. First, any state
  field with a default initializes its `LastValue` channel — so `count: int = 0` really starts
  at `0` — and then your input is applied on top.
- **Reading back:** when the graph settles, every set channel is read into a validated state
  instance and returned from `invoke`.

## Roadmap

Channels are also the natural checkpoint unit: a durable journal (a later phase) snapshots
channel values at each barrier so a run can resume — or fork — from any superstep. Custom
channel types register as plugins.
