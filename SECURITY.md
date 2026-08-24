# Security policy

## Supported versions

The latest release on `main` is supported. The project ships updates roughly
bi-weekly; older tags receive no backported fixes unless a `support/` branch
exists for them (see `BRANCHING.md`).

## Reporting a vulnerability

Please do NOT open a public issue for anything security-sensitive.

Use GitHub's private vulnerability reporting: the **Security** tab of this
repository -> **Report a vulnerability**. Reports go only to the maintainers.
Include what you found, a reproduction, and the impact you believe it has.
You will get an acknowledgement within 7 days.

For everything non-sensitive (crashes, wrong physics, misbehaving sensors),
a normal public issue is the right place.

## Scope worth knowing

This is a simulation library: it opens no network sockets, runs no daemons,
and requires no privileges. The realistic attack surface is that of any
Python package -- code executed at import time and files written by the
recorder (`runs/`, explicit paths only). The `bridge/` module produces
hardware-schema frames but performs no I/O itself.

One boundary deserves emphasis: **simulator output must never be the basis
for a real-world safety decision** (whether an atmosphere is breathable,
whether a leak is present). That is stated in `NOTICE` and
`docs/LICENSES_AND_PROVENANCE.md`, and no fix to this repository can make
such use safe.
