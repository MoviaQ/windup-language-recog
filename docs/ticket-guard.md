# Ticket guard (optional support-ticket veto)

`predict_guarded_ticket` is an opt-in library method for support tickets.
It adds one sentence-level veto on top of the normal whole-ticket vote and
can return `und` (withhold) when a single sentence confidently disagrees
with the rest. It never changes the default APIs, CLI, weights, or
checkpoint: `predict`, `predict_many`, `predict_ticket`, and
`predict_details` behave exactly as before.

## Usage

```python
from language_detector import LanguageDetector

detector = LanguageDetector("model.pt")
label = detector.predict_guarded_ticket(
    title="Login problem",
    description="Nie mogę się zalogować po zmianie hasła.",
    cutoff=0.65,
)
if label == "und":
    # Withheld: route to an explicit fallback you own (manual review,
    # a clarifying prompt, another check). The library makes no network
    # calls and sends nothing anywhere; integrate your own fallback,
    # not an automatic remote API, key, or service.
    ...
```

`und` means "withheld for lack of trustworthy evidence", not a language.
Handle it explicitly in your application; do not silently map it to a
default language.

## How it decides

1. The title and description go through the same shared cleanup as the
   whole-ticket text (`ticket_text.py`): the description is preferred when
   substantive, otherwise the title — exactly the same field the plain
   whole-ticket vote sees. An unused foreign subject line therefore cannot
   veto a request written in another language, and removed code, mail
   quotes, and sign-offs never become evidence.
2. The whole-ticket vote uses the existing calibration: confidence,
   agreement, language/length cutoffs, and margin floors from
   `predict_details`. No new training, no new thresholds.
3. Each remaining sentence with at least 3 Unicode letter words is scored
   with the same raw model (`best_language` plus raw confidence). Short
   1–2 word requests stay in the population but never vote as sentences.
   Digit/underscore tokens (IDs, versions) do not count as words.
4. If any sentence predicts a different language with raw confidence at
   or above `cutoff` (default 0.65), the answer is `und`. The cutoff is a
   raw probability comparison, not an accuracy guarantee. Uncertain and
   lexical add-ons stay disabled. Letterless input returns `und`; other
   model errors propagate.

Practical consequences: plain quoted English error lines are conservative
`und` candidates, fenced code blocks are ignored, and mixed text without a
clear confident foreign sentence can still answer — the guard can return
`und` when it finds evidence, but it is not a reliable detector of all
mixtures. Only the first 2,000 normalized characters (the model-visible
prefix) can supply sentences. Each ticket costs the whole-ticket pass
plus one pass per sentence; no latency or memory figure is claimed here.

No fastText runs at inference time. The runtime needs only the bundled
weights plus `torch` and `ftfy`.

## Measured check (small, not production proof)

The committed fixture `tests/fixtures/authored_tickets.jsonl` holds 84
AI-authored support tickets with distinct family identifiers and own,
human-unverified labels (MIT-licensed; 38,454 bytes; input SHA-256
`1c2f975bb4eb677ddb23b644e4e62c9bc8c2d2378a03e767e23ef4ba5daa124b`).
No real user, client, or ticket text is in it, and no third-party corpus
text, boss-84 rows, or retrained weights were imported for it. Its labels
are author intent frozen before any model ran. The fixture is directional
only: distinct family identifiers do not establish statistical independence,
and these examples cannot prove production behavior. The earlier development
pool had 200 drafts in 53 correlated families; it is not this 84-row fixture.
Nothing here claims the guard beats fastText in general.

Audit note: after the original dataset freeze, an audit found two generic
request titles repeated in development (`holdout-002`, `holdout-057`);
their bodies and selected
model inputs are distinct, and the conservative 82-row cohort below
excludes them. This is an observed overlap note, not a preregistered
clean-holdout claim — the 82-row view is an observer check, not an
independent human benchmark.

Confirmed counts on the release code (no fitting on the fixture): C means
correct accepted answers, E means wrong accepted answers, and U means all
withheld answers. A desired abstention on a mixed example is U, never C.

| cohort | system | C | E | U |
|---|---|---|---|---|
| 84 rows | whole-ticket vote (`main_policy`) | 70 | 5 | 9 |
| 84 rows | guard (`main_guarded`) | 70 | 0 | 14 |
| 84 rows | fastText full 0.65 + sent 0.65 | 68 | 0 | 16 |
| 84 rows | fastText full 0.80 + sent 0.65 | 67 | 0 | 17 |
| exact-coverage matched, N=70 | guard | 70 | 0 | 14 |
| exact-coverage matched, N=70 | fastText (top full scores after the same guard, no gold) | 69 | 1 | 14 |
| conservative 82 rows | guard | 68 | 0 | 14 |
| conservative 82 rows | fastText 0.65 | 67 | 0 | 15 |
| conservative 82, exact-matched 68 answers (tie) | both | 68 | 0 | 14 |

The guard's 14 withheld rows are 13 desired mixed abstentions plus 1 pure
withheld; the whole-ticket vote's 5 errors are mixed false acceptances.
DEV context (not a production promise): the frozen guard was a DEV
non-winner at 53/1 vs fastText 57/1.

The guard's sentence scores are pre-acceptance scores: the raw top-class
probability read before any abstention, including the stored temperature.
"Raw confidence" is not an accuracy guarantee, and fastText scores are
not cross-calibrated with ours — compare counts, never raw numbers
across systems.

Limits to keep in mind: the guard can miss CJK or very short mixed text
with no clear confident segment, so a mixed ticket may still answer —
`und` happens when the guard finds evidence, not for every mixture.
Shared cleanup treats fenced/multiline blocks safely with a fallback to
the cleaned whole text; the guard always reads the same chosen block as
the whole vote. Each ticket costs the whole-ticket pass plus one pass
per sentence; the benchmark records wall seconds for its own run only
(single hardware-dependent sample, not a latency claim).

## Reproduce it

From the repository root:

```bash
./setup.sh
.venv/bin/python -m pip install -r requirements-benchmark.txt
mkdir -p reports
curl --fail --location --output reports/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
sha256sum reports/lid.176.bin
.venv/bin/python tools/benchmark_ticket_guard.py --input tests/fixtures/authored_tickets.jsonl --model model.pt --ft-model reports/lid.176.bin --out reports/ticket-guard-benchmark.json
.venv/bin/python -m unittest discover -s tests
```

Notes on the commands above:

- `requirements-benchmark.txt` pins `fasttext-wheel==0.9.2` with
   `numpy<3` (the tool uses the LIST form `model.predict([text], k=1)`,
  tested against that binding). It is benchmark-only: the runtime stays
  `torch` + `ftfy` from `requirements.txt`, and the guard never imports
  fastText at inference time.
- `reports/` is untracked workspace for the downloaded weights and the
   JSON output; it is not part of the release.
- The separately downloaded `lid.176.bin` weights use **CC BY-SA 3.0**;
  see the [official fastText model documentation](https://fasttext.cc/docs/en/language-identification.html).
  They are not bundled with this repository or used by the runtime guard.
  Observe their attribution and weight-license terms before use or redistribution.
- `sha256sum reports/lid.176.bin` must print the locally verified
  official file fingerprint
  `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e`.
  No upstream cryptographic checksum is claimed; this is the hash of the
  file the benchmark actually ran against.
- CLI flags (confirmed against `--help`, which works without the
  fastText import): `--input`, `--model`, and required `--ft-model` take
  explicit local weight/fixture paths; `--threads` defaults to 2;
  `--out` optionally writes the JSON report (otherwise stdout).
- Provenance recorded in the report: model SHA-256
  (`358586ac346be2f4…`, unchanged v21 weights), input SHA-256 above,
  fastText file fingerprint, Python 3.12.3 / torch 2.14.1+cpu as run,
  sentence cutoff 0.65, and `fitted_on_input: false`.
- This changes no checkpoint: still the bundled v21 weights, still an
  unreleased software feature — no new version or v22 mark.
