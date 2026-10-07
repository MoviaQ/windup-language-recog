# Security

Inference runs locally and does not upload input text. Dataset preparation makes explicit downloads from the public sources documented in `prepare_public_data.py`. Runtime dependency installation contacts package indexes.

Load checkpoints only from sources you trust. The loader uses PyTorch's `weights_only=True`, but this is not a substitute for trusted artifacts, current dependencies, or resource limits. Bound input size and batch size when exposing inference through your own service. Input cleanup is designed for language classification; it is not a privacy redactor or HTML security sanitizer.

For security-sensitive reports, use the repository's GitHub **Security → Report a vulnerability** feature if enabled. Otherwise contact the repository owner through GitHub before sharing exploit details. Never include credentials, private text, or a working exploit against a third-party system in a public issue. Ordinary prediction errors belong in the bug-report template.

Version 0.1.x is the initial maintained series; no response-time or service-level guarantee is offered.
