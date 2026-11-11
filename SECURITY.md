# Security Policy

## Supported versions

Security fixes go into the latest release on [PyPI](https://pypi.org/project/h2ml/).
Older versions are not patched; upgrade to the latest release.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub:
[Report a vulnerability](https://github.com/h2ugoparra/h2ml/security/advisories/new)
(the **Security** tab of the repository, then **Report a vulnerability**).

Include the h2ml version, what the problem allows, and steps to reproduce it. You
should get a reply within a week. Once a fix is released, the advisory is published
and you are credited unless you ask not to be.

## Loading saved results and models

`PipelineResult.load()`, `MultiMetricResult.load()`, `FinalModel.load()` and
`DeltaFinalModel.load()` read pickle files (via `joblib`). Unpickling runs code from
the file, so only load results and models you created or that come from a source you
trust.
