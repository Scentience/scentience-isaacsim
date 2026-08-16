# Sim-to-real path (v0.1 summary)

1. Every coefficient carries an evidence level (provenance.py): MEASURED >
   DATASHEET > DIGITIZED > SYNTHESIZED > ASSUMED. `claim_check()` refuses
   statements the evidence cannot support; scope claims with `depends_on`.
2. Shipping synthesized/digitized coefficients is fine BECAUSE they are
   labelled; replace incrementally as bench data lands (roadmap v0.3).
3. Calibration protocol (when data arrives): split by session AND device
   (never randomly -- temporal leakage), fit static (A, beta) then temporal
   (tau_rise, tau_fall, dead volume), evaluate on held-out DEVICES,
   replay held-out exposure sequences through the sim, report per-channel
   RMSE / correlation / lag / rise-recovery error.
4. Domain randomisation is the bridge until then: R0 log-uniform over the
   datasheet spread, (A, beta, tau, drift) jittered per episode; policies
   consume drift-invariant features only (deflection, derivative, EMA).
5. The whiff-retention number (19% packaged vs 97% fast) is the single
   most consequential sim-to-real parameter: state the sensor profile in
   every result.
