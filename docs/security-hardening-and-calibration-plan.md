# Crucible — Security Hardening & Calibration Benchmark: Execution Plan

## Context

**Why this work exists.** Crucible is a self-improving coding agent: a Plan→Execute→Test→Reflect loop with
embedding-backed memory, falsifiable self-predictions, a record-only replay engine that scores those
predictions, and a calibration/benchmark harness. Two gaps remain between the current state and a defensible,
production-grade system:

1. **The innovation claim is unproven.** The benchmark harness is **code-complete but never run**, so the
   headline claim — *"the agent's self-predictions are calibrated"* — has no real numbers yet. Running it
   produces the calibration curve that converts an interesting design into a measured result.

2. **The execution boundary is not a security boundary.** Crucible *is* an autonomous agent that executes
   untrusted, model-generated code — a high-risk threat surface. Today the isolation layer says so explicitly
   (`agent/executor/sandbox.py:1-9`, `agent/safety/checker.py:162-170`): the AST safety checker is best-effort
   and bypassable, network access is all-or-nothing, there is no audit trail, and secrets are unmasked. This
   plan replaces "ergonomic isolation" with real, enforced, testable controls.

**Intended outcome.** A repo that ships (a) a measured innovation result and (b) a production-grade agent
security architecture — kernel-isolated execution, policy-as-code, zero-trust egress, tamper-evident
telemetry, and workload identity — documented so it is legible to a security reviewer skimming and defensible
in a deep technical review.

**Key synergy.** Provision the benchmark host (AWS) with production security rigor (IaC, workload IAM, no
static keys, locked-down security group). Run the benchmark there for the innovation numbers. Then re-run the
benchmark *through* the new security layer and report the overhead — quantifying the security-vs-performance-
vs-usability tradeoff with real data instead of assertions.

---

## Security Controls Matrix (threat → control → enforcement → verification)

Coverage is framed the way a security architect reasons about an autonomous agent that executes untrusted,
model-generated code — against recognized taxonomies: **MITRE ATLAS** (adversarial threats to AI systems), the
**OWASP Top 10 for LLM Applications (2025)**, **NIST SP 800-53** control families, and the **CIS Docker
Benchmark**. Each row pairs a threat with an *enforced* control, the exact place it is enforced, and how it is
proven.

| Threat (taxonomy ref) | Control | Enforcement point | Verification |
|---|---|---|---|
| Sandbox escape / untrusted code execution (ATLAS; OWASP LLM05 Improper Output Handling) | Kernel-isolated runtime: gVisor `runsc`, seccomp default-deny, `cap-drop ALL`, non-root UID, read-only rootfs | Docker executor runtime config | Red-team payload contained (`test_security_boundary`) |
| Excessive agency / unbounded actions (OWASP LLM06, LLM10) | Declarative policy: package/path/domain allowlists, resource ceilings, attempt caps | Policy engine at executor/proxy boundary | Disallowed action blocked + logged (`test_policy`) |
| Data exfiltration / C2 over network (ATLAS exfiltration; OWASP LLM02 Sensitive Info Disclosure) | Default-deny egress + allowlist forward proxy; no direct container network | Egress proxy | Denied domain blocked + deny event (`test_egress`) |
| Prompt injection steering the agent (OWASP LLM01) | Controls enforced independent of model intent — no control relies on the LLM "behaving"; model output is adversarial by assumption | Threat model + boundary design | Controls hold regardless of generated code |
| Supply-chain / malicious dependency (ATLAS; OWASP LLM03 Supply Chain) | Package allow/deny + optional version pins; install gated at network *and* policy | DependencyManager + policy + egress | Disallowed install blocked |
| Tampering / repudiation of agent actions (NIST AU; STRIDE T/R) | Tamper-evident, hash-chained, secret-redacted audit log | Audit event bus | Hash-chain verifies; no secrets present |
| Resource exhaustion / DoS (OWASP LLM10 Unbounded Consumption) | mem/cpu/pids limits, circuit breaker, max-iterations | Runtime limits + loop | Limit breach contained |
| Static-secret sprawl / credential theft (NIST AC/IA; CIS) | Workload identity: short-lived role creds, no static keys; SSM-only host access; IMDSv2 required | AWS IAM instance role (Terraform) | No static keys present; SSH ingress closed |
| Weak detection / no monitoring at scale (NIST SI/AU) | Telemetry pipeline aggregating denials/anomalies across *all* executions | Audit → CloudWatch/OTel + `security_report` | Cross-run summary over ~200 runs |

This is the structure a security review expects: defense-in-depth, least privilege, zero-trust egress, and
tamper-evident auditing, each tied to a control point and a test.

---

## Workstreams

- **Track I — Innovation evidence** (Phases 0–3): provision the host, run the benchmark, produce `bench/REPORT.md` + whitepaper.
- **Track S — Security hardening** (Phases 4–6): threat model → hardened runtime → egress/policy → telemetry/identity → red-team.
- **Track N — Documentation** (Phase 7): whitepaper, `SECURITY.md`, controls doc, README polish, design rationale.

Track I and the early design of Track S can proceed together; the Track S build reuses the same host.

---

## Phase 0 — Local prep & baseline (do before touching the cloud host)

Goal: clean branch, green tests, reproducible local smoke so cloud time isn't spent debugging.

1. Branch from current: `git checkout -b track-benchmark-and-security`.
2. Create the **Python 3.12** venv and install deps (sentence-transformers is mandatory or memory recall is a no-op):
   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pip install docker          # executor (commented out in requirements.txt:18)
   ```
3. Run the suite: `pytest -q` (52 tests across `tests/`). Confirm green before proceeding.
4. Build the prebaked image and smoke the runner **locally** (Ollama optional locally; this is just a harness shake-out):
   ```bash
   ./docker/build.sh                      # builds crucible-runtime (docker/Dockerfile)
   python -m bench.runner --smoke         # 3 problems x 2 reps (bench/runner.py:280)
   ```
5. Capture wall-clock per task from `StepProfiler` output to size the cloud instance.

Files of note: `bench/runner.py` (CLI: `--reps`/`--smoke`/`--limit`/`--llm`/`--model`/`--docker-image`/`--out`),
`bench/problems.py` (40 single-function problems), `bench/analyze.py`, `docker/Dockerfile`.

---

## Phase 1 — Provision the benchmark host *as a security exhibit* (Track I + cloud posture)

Goal: a GPU/CPU EC2 box to run the benchmark, provisioned with production rigor. The Terraform itself is the
IaC + workload-IAM + identity evidence — write it cleanly and commit it under `infra/`.

1. **Instance sizing.**
   - Ollama `qwen2.5-coder:7b` is the fixed model (keeps cost out of the calibration analysis).
   - GPU (recommended for 200 runs): `g5.xlarge` (A10G) — fast enough to finish 40×5 in hours.
   - CPU-only fallback: `c7i.4xlarge` — works, slower; fine for smoke + partial.
   - Note: **gVisor (Phase 4) runs on a standard EC2 instance**; Firecracker (stretch) needs a `.metal`
     instance for nested virtualization — defer unless you go that far.
2. **Terraform module `infra/`** (commit it — reviewers read this):
   - EC2 with an **IAM instance profile** (role, not access keys) granting only what's needed (S3 results bucket, SSM).
   - **No SSH ingress**; access via **SSM Session Manager** only. Security group: egress to PyPI/Ollama model
     pull as needed, no public inbound.
   - Encrypted EBS, IMDSv2 required (`http_tokens = "required"`), tags, an S3 bucket (versioned, SSE) for results/artifacts.
   - Document the "no static credentials, short-lived role creds, least-privilege" decisions in `infra/README.md`.
3. Bootstrap (user-data or SSM doc): install Docker, NVIDIA drivers (if GPU), Ollama; `ollama pull qwen2.5-coder:7b`;
   clone repo; build `crucible-runtime`.

**Controls delivered:** workload identity (short-lived role creds, no static keys), least-privilege IAM,
IaC-as-evidence, identity-based host access (SSM not SSH), IMDSv2-required, encryption at rest.

---

## Phase 2 — Run the benchmark (Track I)

Goal: accumulate replay verdicts at scale → raw JSONL.

1. On the instance, in the 3.12 venv with Ollama serving:
   ```bash
   ollama serve &                                   # background
   python -m bench.runner --smoke                   # shake-out on real hardware first
   python -m bench.runner --reps 5 --docker-image crucible-runtime --out bench/results.jsonl
   ```
   - 40 problems × 5 reps ≈ 200 task runs. Memory persists across the whole run (predictions must surface and
     replay across reps) — do **not** reset per problem.
2. Push `bench/results.jsonl` to the S3 bucket (and pull locally for analysis/commit).

---

## Phase 3 — Calibration analysis + innovation writeup (Track I)

Goal: turn JSONL into the headline result and a publishable narrative.

1. `python -m bench.analyze bench/results.jsonl > bench/REPORT.md` (replaces the synthetic placeholder).
2. Read out the metrics `bench/analyze.py` computes:
   - **Calibration-by-confidence bucket** — *the* headline: are higher-confidence self-predictions confirmed more often?
   - **Off-topic rate** — integrity metric; report prominently, never hide (a high rate = thin calibration sample).
   - **Surviving-predictions catalog** — concrete antipatterns that confirmed (e.g. `[]` → IndexError).
   - **Convergence variance** — does iterations-to-success tighten as memory fills?
   - **Memory-helped** correlation; **retirement stats** (note it's a cross-run mechanism; likely "0 retired this run, by design").
3. Honesty guardrails for the claim: the thesis is the narrow, true one — *"calibrated self-prediction"* — not "AGI."
   If the curve is flat/noisy, that is still a publishable negative result; report it straight.
4. Draft `docs/WHITEPAPER.md` (the innovation case) and an external write-up from the report.

---

## Phase 4 — Threat model + hardened runtime (Track S, the core security work)

Goal: replace "ergonomic isolation" with a real, enforced boundary; document the adversary first.

1. **`SECURITY.md` threat model** (design first):
   - Adversary = the **LLM-generated code** and **prompt-injection** into the agent; trust boundaries drawn around
     the executor and the network. STRIDE pass. State assumptions (PyPI/LLM endpoint trusted) explicitly.
   - Honestly catalog today's gaps (AST bypass via aliasing/`getattr`, all-or-nothing network, no audit, secrets
     unmasked) — then show what each Phase-S deliverable closes.
2. **Harden the Docker executor** (`agent/executor/docker_executor.py`):
   - gVisor runtime (`--runtime=runsc`) as the kernel-isolation boundary; document install on the EC2 host.
   - seccomp profile (default-deny syscalls + allowlist), `--cap-drop=ALL`, `--security-opt=no-new-privileges`,
     **non-root UID**, **read-only rootfs** + tmpfs scratch, `--pids-limit`, enforce `mem_limit`/`nano_cpus`.
   - Keep the local subprocess sandbox but **gate it behind an explicit `--unsafe-local` flag** and label it
     non-isolating in code + docs (it already says so; make it un-bypassable as the default).
3. **Adversarial red-team test suite** `tests/test_security_boundary.py`:
   - First *prove* the current AST checker is bypassable (aliasing, `getattr(__builtins__,...)`, `__import__`).
   - Then prove the hardened runtime contains those same payloads (no FS escape, no raw network, no priv-esc).
   - This pair (attack → contained) is the most persuasive artifact in the security story.

**Controls delivered:** kernel-isolated sandbox (gVisor), container hardening (CIS-aligned), defense against
untrusted model output (ATLAS / OWASP LLM05), demonstrated escape-containment.

---

## Phase 5 — Network egress control + policy engine (Track S)

Goal: move from "network on/off" to identity/policy-based, enforced at the boundary — not best-effort AST.

1. **Egress allowlist proxy** `agent/security/egress_proxy.py` (+ wiring in `docker_executor.py`):
   - Container gets **no direct network** (`network_disabled` / internal-only network); all egress routed through a
     forward proxy that allows only a configured domain allowlist (PyPI mirror + the LLM endpoint). Everything
     else denied and logged.
   - This finally makes `DependencyManager` installs (`agent/dependency_manager.py`) policy-gated at the network
     layer, not just by an interactive confirm.
2. **Declarative policy engine** `agent/security/policy.py` + `policy.yaml`:
   - One enforced policy object: allowed install packages (allow/deny list, optional version pins), writable paths,
     reachable domains, resource ceilings, max install attempts. Enforced at the executor/proxy boundary.
   - Reframe `agent/safety/checker.py` honestly as **defense-in-depth signal, not the boundary**; the policy +
     runtime are the boundary. Decisions emit to the telemetry pipeline (Phase 6).
3. Tests: `tests/test_policy.py`, `tests/test_egress.py` — denied package, denied domain, path-traversal attempt
   (extend the existing `_resolve_workspace_path` check at `docker_executor.py:265`), resource-ceiling breach.

**Controls delivered:** zero-trust egress (default-deny + allowlist), policy-as-code enforcement,
supply-chain gating (OWASP LLM03), excessive-agency containment (OWASP LLM06).

---

## Phase 6 — Safety-monitoring telemetry pipeline + workload identity (Track S)

Goal: harden safety-monitoring across agent executions at scale, end to end.

1. **Structured audit event bus** `agent/security/audit.py`:
   - Every security-relevant action emits a structured JSON event: code executed (hash), file writes, package
     installs, network attempts (allow/deny + domain), policy decisions, resource-limit hits, replay/circuit-breaker
     events. Stable schema, monotonically ordered, **secret-redacted** (mask API keys in stdout/stderr/state).
   - Make it **tamper-evident**: hash-chain the event log (each event carries the prior event's hash).
2. **Pipeline at scale:** ship events to CloudWatch Logs / OpenTelemetry collector on the host; provide a small
   query/aggregation script (`bench/security_report.py`) that summarizes denials, anomalies, and per-task action
   counts across the whole benchmark run — i.e. monitoring *across executions at scale*, not one run.
3. **Per-task workload identity:** issue scoped, short-lived credentials per task via the AWS instance role
   (IRSA-style pattern, no static keys); document how identity federation would extend this to a fleet.
4. **Security/performance/usability measurement:** re-run `python -m bench.runner --reps 5` *through the hardened
   runtime + egress proxy + policy + audit* and add an **overhead table** (latency, success-rate delta) to the report.
   This quantifies the security-vs-performance-vs-usability tradeoff with real data.

**Controls delivered:** tamper-evident audit (NIST AU), monitoring-at-scale (NIST SI), workload identity /
federation pattern, quantified security-vs-performance tradeoff.

---

## Phase 7 — Documentation & write-up (Track N)

Goal: make both the innovation result and the security architecture legible to a reviewer skimming and
defensible in a deep technical review.

1. `docs/WHITEPAPER.md` — innovation case (Phase 3 result), 1–2 pages, with the calibration figure.
2. `SECURITY.md` — threat model + the controls map (Phases 4–6), with a before/after table and the overhead numbers.
3. `docs/SECURITY-CONTROLS.md` — the controls matrix above as a standalone security document: each
   threat→control→enforcement→verification row linked to the implementing file and its test, with the framework
   refs (MITRE ATLAS / OWASP LLM Top 10 / NIST families / CIS Docker). Honest status per row ("enforced & tested"
   vs. "designed, not yet run").
4. README polish — lead with the two theses; add a "Security architecture" section and an architecture diagram.
5. `docs/DESIGN-RATIONALE.md` — design-tradeoff FAQ: Why gVisor over seccomp-only? Why an egress proxy over
   `network_disabled`? Where does the AST checker still add value? How would this scale to a fleet? What did the
   overhead cost? Plus the "what I'd do next at production scale" roadmap.
6. External write-up from the whitepaper + a short security post from `SECURITY.md` (publishing best practices).

---

## Sequence summary (operations order)

1. **Phase 0** local: branch, 3.12 venv, `pytest -q`, build image, `--smoke`.
2. **Phase 1** host: Terraform (`infra/`) → secure EC2 (IAM role, SSM, IMDSv2, encrypted EBS, S3 results).
3. **Phase 2** run: `--smoke` on hardware, then `--reps 5 --out bench/results.jsonl`; push to S3.
4. **Phase 3** analyze: `bench/analyze.py` → `bench/REPORT.md`; draft `docs/WHITEPAPER.md`.
5. **Phase 4** security: `SECURITY.md` threat model → gVisor/seccomp/caps/ro-rootfs runtime → red-team tests.
6. **Phase 5** network/policy: egress allowlist proxy → `policy.yaml` engine → policy/egress tests.
7. **Phase 6** telemetry/identity: hash-chained audit bus → CloudWatch/OTel pipeline → workload identity →
   re-run benchmark hardened → overhead table.
8. **Phase 7** docs: whitepaper, `SECURITY.md`, `SECURITY-CONTROLS.md`, README, design rationale, external write-up.

> Minimum viable initial release if time is short: **Phases 0–4 + 7** (real numbers + real isolation + threat
> model + docs). Phases 5–6 are the differentiators that take it to a production-grade security posture.

---

## Verification

- **Tests green throughout:** `pytest -q` after each Track-S phase; new suites
  (`test_security_boundary.py`, `test_policy.py`, `test_egress.py`) must pass, and the red-team tests must show
  *attack contained* (not just "code runs").
- **Innovation result real:** `bench/REPORT.md` regenerated from `bench/results.jsonl` (not the synthetic
  placeholder); off-topic rate reported; claim scoped to "calibrated self-prediction."
- **Isolation actually holds:** demonstrate a payload that bypasses `agent/safety/checker.py` AST analysis, then
  show the same payload contained by the gVisor/seccomp boundary (no FS escape, no raw egress, non-root).
- **Egress/policy enforced:** a denied package and a denied domain are blocked *and* appear as deny events in the
  audit log; a path-traversal attempt is rejected.
- **Telemetry at scale:** `bench/security_report.py` summarizes denials/anomalies across all ~200 runs; audit log
  hash-chain verifies; no secrets present in logs/state.
- **Tradeoff quantified:** overhead table (hardened vs. baseline latency + success-rate delta) present in the report.
- **Cloud rigor reviewable:** `infra/` Terraform plans cleanly; no static keys; SSM-only access; IMDSv2 required.
- **Documentation complete:** WHITEPAPER, SECURITY, SECURITY-CONTROLS, README, DESIGN-RATIONALE all present and
  internally consistent; controls matrix rows each resolve to a real file + passing test.
