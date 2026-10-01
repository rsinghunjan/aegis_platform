# Model Card Template

Use this template as MODEL_CARD.md in each model directory. Fill in fields with concise, accurate information.

Model Details
- Name: <model name>
- Version: <version>
- Description: Brief description of what the model does and intended usage.

Model Owner & Contacts
- Owner: <team or person>
- Contact: <email or slack>

Intended use
- Primary intended tasks
- Allowed use cases
- Prohibited use cases

Datatypes & Inputs
- Modalities (text/image/audio)
- Expected input formats and preprocessing

Training Data
- Dataset manifest: <path or URL>
- Dataset license(s)
- Dataset description and known biases

Evaluation
- Metrics: include major metrics (e.g., accuracy, F1) and evaluation protocol
- Test datasets used and their provenance

Performance & Limitations
- Known failure modes
- Expected accuracy ranges and recommended confidence thresholds

Privacy & Safety
- Does the model store or leak PII? <yes/no>
- Any mitigation used (de-identification, DP)
- Safety checker version / policy used during inference

Reproducibility
- Training code source: <git url + commit>
- Training hash or run id: <training_hash or MLflow run id>
- How to reproduce training (short steps)

Artifact
- Artifact path: <artifact file name, DVC path or MLflow artifact URI>
- How to load the model (example code snippet)

License
- License: <SPDX id or license text>

Changelog
- v1.0 - initial release

