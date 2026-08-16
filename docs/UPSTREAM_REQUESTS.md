# Requests for the Scentience client packages

Findings from integrating against `scentience` (PyPI) v2.2.2, for Kordel.
None block this release; all would tighten the hardware/sim loop.

1. **Sockets API client is not in the PyPI package.** v2.2.2 ships the BLE
   client (`ScentienceDevice`, bleak-based) and the OVL client only. The
   documented Sockets API (host:9000, `sample()`/`stream()`) has no Python
   client in the package. If it lands there (and in the C++/Rust/NPM
   siblings), this simulator can serve that wire protocol directly and every
   client library would work against the sim unchanged -- true
   hardware-in-the-loop symmetry, zero adapter code for users.
2. **Units are not declared anywhere in the BLE schema.** The 14 compound
   channels carry no ppm/ppb declaration in the docs or the client. The sim
   emits `_sim_units: "ppm"` as an extension key; a `UNITS` field in the real
   schema (or a docs statement) would remove the ambiguity for everyone.
3. **`device_reading_to_ovl` channel mismatch.** The client itself warns the
   device channels do not align 1:1 with the OVL reference array (no Alcohol
   channel; LPG approximated). For the release-2 COLIP integration, a
   versioned channel-mapping table in the package would let sim and hardware
   share one mapping.
4. **BLE UUIDs are placeholders in the docs.** Not sim-relevant, but it
   blocks anyone writing an independent BLE client.
