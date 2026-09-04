# Recovered Training Environment

The exact server-level software and hardware record is frozen in
`training_environment.txt`; the complete package listing is in
`pip-freeze-training-server.txt`. `requirements.txt` remains a portable package
family specification rather than an exact lockfile.

The experiment server directory did not contain `.git`, so the historical
training commit is unavailable. This release does not invent one. Instead,
`source_hashes.sha256` records the server-matched LPR implementation, trainer,
runner, and analyzer hashes, while the release-wide `SHA256SUMS.txt` covers
every distributed file. The local repository HEAD used during release
assembly is recorded separately and must not be interpreted as the historical
training commit.
