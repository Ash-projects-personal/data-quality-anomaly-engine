# How this works, in plain words

## Why two stages

Rules catch the errors you can describe exactly: a quantity that
is negative, a price that is zero, a missing customer ID. They
are perfect on precision here because a negative quantity in this
test is always a corruption by construction. But rules cannot
catch a line that is individually legal and jointly strange, like
5,000 units at one cent. That is the forest's job: it learns the
shape of normal lines across quantity, price and line total
together, and flags lines that sit far from that shape.

Neither stage is enough alone, and now I can say that with
numbers: rules recall 0.25, forest recall 0.718, combined 0.968.

## The labelled test, and its limits

Injecting corruptions into real data is a compromise. The clean
rows are genuinely real, with real price and quantity patterns,
which a generator would not reproduce faithfully. The corruptions
are mine, so the detector is graded on the kinds of errors I
thought to inject. A corruption type I did not imagine is not in
the test, and real world recall will be lower than 0.968 for
exactly that reason. What the test does prove is narrower and
still useful: on these three error types, in real transaction
data, the combined detector catches almost everything the rules
and the forest each miss alone.

## The clipping detail

The forest's features are log(1 + x) with negatives clipped to
zero. I chose that so huge real orders would not dominate the
scale. A side effect is that negative quantities collapse to the
same point as zero quantities, and the forest cannot see them.
I could drop the clip and let the forest see negatives, but then
it would just learn a rule the rules already state exactly.
Knowing which stage is blind to what is part of running a
detector responsibly, and it is written down here so I can
explain it in an interview instead of discovering it there.
