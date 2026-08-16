import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from scentience_olfaction.provenance import (Evidence, ProvenanceRegistry,
                                             coeff, synthesized_from)


def make_reg():
    r = ProvenanceRegistry()
    r.register("plume.gamma", coeff(2e-3, Evidence.ASSUMED, "modelling choice"))
    r.register("mox.A.ethanol", coeff(1.31, Evidence.DIGITIZED, "driver fit"))
    r.register("mox.tau.fast", coeff(0.046, Evidence.MEASURED,
                                     "Dennler 2024", n_units=8))
    return r


def test_claim_scoping_pass_and_fail():
    r = make_reg()
    ok, why = r.claim_check("fast tau is measured", Evidence.MEASURED,
                            depends_on="mox.tau.")
    assert ok, why
    ok, why = r.claim_check("absolute ppm accuracy", Evidence.MEASURED,
                            depends_on="mox.")
    assert not ok and "DIGITIZED" in why
    ok, why = r.claim_check("anything", Evidence.ASSUMED, depends_on="nope.")
    assert not ok and "no registered coefficients" in why


def test_report_and_json_roundtrip(tmp_path):
    r = make_reg()
    rep = r.report()
    assert "ASSUMED" in rep and "plume.gamma" in rep
    p = tmp_path / "prov.json"
    r.to_json(p)
    import json
    d = json.loads(p.read_text())
    assert d["mox.tau.fast"]["evidence"] == "MEASURED"
    assert d["mox.tau.fast"]["n_units"] == 8


def test_synthesized_source_names_donor():
    s = synthesized_from("ethanol", "h2s", "same die")
    assert "ethanol" in s and "h2s" in s


def test_weakest_and_device_registration():
    r = make_reg()
    assert r.weakest() == Evidence.ASSUMED
    from scentience_olfaction.sensors.scentience_v1 import register_coefficients
    r2 = register_coefficients(ProvenanceRegistry())
    assert len(r2.entries) >= 8
    ok, _ = r2.claim_check("device ppm is calibrated", Evidence.MEASURED,
                           depends_on="mics6814.")
    assert not ok, "shipping coefficients must NOT support a MEASURED claim"
