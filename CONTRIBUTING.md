# Contributing

Thank you for helping make language detection understandable and useful.

## Development setup

```bash
./setup.sh
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python -m unittest discover -s tests
```

Inference changes need only runtime dependencies; install training dependencies when working on datasets, training, or evaluation. See [training](docs/training.md) for a public corpus reproduction.

## Changes and pull requests

Create a branch with a descriptive name, keep changes focused, and explain the user-visible behavior in your pull request. Include commands used for verification and any known limitations. For inference or preprocessing changes, add behavior tests that exercise actual cases; for model changes, report dataset provenance, split separation, and reproducible evaluation settings.

Use readable Python, descriptive names, type hints where they clarify interfaces, and small functions. Follow existing formatting. Preserve `LanguageDetector` API behavior and compatibility with checkpoint configuration defaults. Do not commit source archives, prepared corpora, feature caches, credentials, or machine-specific files. Avoid replacing the release checkpoint casually: a new model needs a documented provenance and benchmark.

## Main branch protection

The `main` branch is protected by active GitHub rulesets:

- All changes must go through a pull request; direct pushes are blocked, including for the owner.
- Both GitHub Actions checks, `test (3.10)` and `test (3.12)`, must pass before merging. These checks cannot be bypassed by the owner.
- Only repository administrators can merge pull requests. Currently, `MoviaQ` is the sole administrator; granting another account administrator access would also grant it merge permission.
- Approval from another reviewer is not required.
- Force pushes and deletion of `main` are blocked.

These rules are enforced in GitHub, not by this document. Administrators can inspect them in the [repository ruleset settings](https://github.com/MoviaQ/windup-language-recog/settings/rules).

## Issues

Include expected and actual behavior, Python/PyTorch versions, operating system, the API or command used, and a minimal synthetic text that reproduces the issue. Do not include personal information or confidential input text. Feature requests should explain a concrete use case and the desired behavior.

## Using Codex

Open this repository in Codex and read [AGENTS.md](AGENTS.md). Ask for a focused change and run the documented tests. Review generated code and factual claims before submission, especially preprocessing rules, training splits, and performance numbers.

## Licensing

By contributing, you agree that your software, documentation, authored examples, and artwork contributions use MIT. Model weight contributions use the terms in [MODEL_LICENSE.md](MODEL_LICENSE.md); include the provenance and upstream rights needed for any proposed model or data change. Do not assume a repository license relicenses third-party data.
