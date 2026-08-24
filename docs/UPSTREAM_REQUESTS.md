# Upstream requests

Changes this simulator needs from **other** Scentience components. Each entry
states the gap, the workaround shipped here, and what would let the workaround
be deleted.

These are tracked here rather than worked around silently, because a workaround
nobody wrote down becomes permanent.

---

## UR-1: The BLE schema publishes no units

**Component**: `scentience` client packages (PyPI, and the C++/Rust/NPM
siblings) and the device firmware BLE API.
**Status**: open as of v0.1.
**Severity**: high -- it is a correctness hazard, not a convenience issue.

### The gap

`ScentienceDevice.sample_ble()` emits compound magnitudes (`CO2`, `NH3`,
`C2H5OH`, ...) as bare numbers with **no unit field anywhere in the frame**. A
consumer cannot tell ppm from ppb from a raw ADC count from a fractional
`Rs/R0` deflection. Today that is resolved by out-of-band convention, which is
to say by tribal knowledge.

This matters most at exactly the sim-to-real boundary this package exists to
serve. The entire design goal of `bridge/ble_schema.py` is that code written
against hardware consumes simulator output unchanged. If units are conventional
rather than declared, "unchanged" is doing unearned work: a silent
ppm-vs-ppb mismatch is a factor of 1000 that no type system will catch.

### Workaround shipped here

`bridge/ble_schema.py` emits **ppm by volume** and declares it in the frame
under a `_sim_units` key:

```python
frame["_sim_units"] = "ppm"
```

The leading underscore marks it as an **extension key that hardware does not
send**. Consumers must tolerate its presence and must not depend on it when
reading real devices. It exists so that simulator output is self-describing
even though hardware output is not.

### What would let us delete it

A `units` field in the BLE schema proper -- per-compound, or one frame-level
declaration if all compounds share a unit. Then simulator and hardware frames
become genuinely identical and `_sim_units` goes away.

Until then, `_sim_units` stays, and it stays underscore-prefixed so nobody
mistakes it for a hardware field.

---

## UR-2: No Sockets API protocol in the client packages

**Component**: `scentience` client packages.
**Status**: blocked -- deliberately not implemented here.
**Target**: v0.5 (see `ROADMAP.md`).

### The gap

There is no published wire protocol for streaming device frames over a socket.
A Sockets-API server in this simulator would let a policy or analytics stack
consume simulated devices over the network exactly as it consumes real ones --
useful for hardware-in-the-loop and for language/runtime boundaries where
importing a Python package is not an option.

### Why nothing is shipped

Inventing a protocol here would be **defining the standard by accident**. The
client packages are the right place for the protocol to be specified; this
simulator should implement it, not author it. Shipping a unilateral wire format
would either fragment the ecosystem or silently become the de facto standard
without review -- and this package's whole positioning is that a standard
should be deliberate.

### What would unblock it

A specified protocol in the client packages: framing, handshake, versioning,
and error semantics. This repo will implement the server side against that
spec.

---

## UR-3: No published tabulated MiCS-6814 coefficients

**Component**: upstream sensor vendor (SGX Sensortech), not Scentience.
**Status**: open, and probably permanent.
**Severity**: medium -- bounded and disclosed, but it caps what can be claimed.

### The gap

The MiCS-6814 datasheet (rev 8) publishes sensitivity as **log-log graphs
only**. There is no tabulated `(A, beta)` for the power law
`Rs/R0 = A * C^-beta`, and for some analytes on the RED curve (notably H2S)
there is no tabulated range at all.

### Workaround shipped here

Coefficients are algebraically inverted from open-source driver fits, which
were themselves digitized from those graphs. They carry evidence level
`DIGITIZED`; H2S is `SYNTHESIZED` by stated analogy from ethanol on the same
RED die.

`provenance.py` enforces the consequence: `claim_check()` **refuses** any
absolute-ppm accuracy claim resting on these coefficients. Run
`python scripts/provenance_demo.py` to watch it decline one.

### What would let us delete it

Per-unit calibration against a reference instrument on Scentience hardware,
promoting these to `MEASURED` with an `n` and an uncertainty. That is a
hardware-lab task, not a code task, and it is the single highest-value
provenance upgrade available to this project.

---

## Filing a new entry

Add a section with an `UR-n` identifier, the owning component, the gap, the
workaround shipped here, and the concrete condition that would remove it. If a
workaround lands in code, the code comment must point back at this file -- as
`bridge/ble_schema.py` does for UR-1.
