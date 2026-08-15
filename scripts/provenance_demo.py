import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scentience_olfaction.provenance import (
    ProvenanceRegistry, Evidence, coeff, synthesized_from)

r = ProvenanceRegistry()
# Realistic mix for a first release: some measured, most not.
r.register("mics6814.red.tau_rise", coeff(
    0.29, Evidence.MEASURED, "Scentience bench 2026-07, step exposure 0->100 ppm EtOH",
    units="s", conditions="22 C, 45 %RH, 0.5 m/s face velocity", n_units=6, uncertainty=0.04))
r.register("mics6814.red.R0_range", coeff(
    4.0e5, Evidence.DATASHEET, "SGX MiCS-6814 rev 8, RED die",
    units="ohm", notes="100k-1.5M spread across units; sample log-uniform"))
r.register("mics6814.red.A.ethanol", coeff(
    1.31, Evidence.DIGITIZED, "inverted from open-source driver fit C=1.52*r^-1.55",
    notes="datasheet publishes log-log graphs only; no tabulated A/beta exists"))
r.register("mics6814.red.beta.ethanol", coeff(0.645, Evidence.DIGITIZED, "as above"))
r.register("mics6814.red.A.hydrogen_sulfide", coeff(
    2.0, Evidence.SYNTHESIZED,
    synthesized_from("ethanol", "hydrogen_sulfide", "same RED die, similar reducing-gas beta"),
    notes="H2S appears on the datasheet RED curve but NO range is tabulated"))
r.register("plume.gamma", coeff(
    2.0e-3, Evidence.ASSUMED, "modelling choice", units="m^2/s",
    notes="Farrell reports 1e-3; tuned up for indoor scale. Not measured."))

print(r.report())
print("\n" + "=" * 70)
for claim, lvl in [
    ("simulated MiCS-6814 response is accurate to within 10% in absolute ppm", Evidence.MEASURED),
    ("the framework reproduces published turbulent plume statistics", Evidence.DIGITIZED),
]:
    ok, why = r.claim_check(claim, lvl)
    print(f"\n[{'OK ' if ok else 'NO '}] {why}")
