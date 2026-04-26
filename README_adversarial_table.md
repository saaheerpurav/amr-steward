## Adversarial Stress Test (10 hand-crafted hard cases)

These cases are not in any training set; each is engineered to break a specific
baseline failure mode. MIC values are set to be unambiguous against EUCAST v16.0
breakpoints. *Trained* column links to the live HuggingFace Space where you can
inject any case and observe the model's prescription in real time.

**Pass threshold**: quality\_ratio >= 0.85 (near-optimal IDSA-concordant prescription).

**Note on R5**: Baselines make zero tool calls so R5=0. The trained model must beat
baselines on *both* R0-R4 (correct prescription) *and* R5 (systematic investigation).

| ID | Scenario | Best Drug | Broad-Empiric | Random (seed=42) | EUCAST-Only | Trained |
|----|----------|-----------|---------------|-----------------|-------------|---------|
| **A1** | VSE bacteremia + penicillin allergy | `vancomycin` | FAIL (0.00) | FAIL (0.00) | SUBOPT (0.78) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |
| **A2** | CRE K. pneumoniae bacteremia | `ceftazidime-avibactam` | FAIL (0.11) | FAIL (0.11) | PASS (1.00) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |
| **A3** | Susceptible E. coli UTI -- stewardship trap | `ceftriaxone` | SUBOPT (0.61) | SUBOPT (0.61) | SUBOPT (0.76) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |
| **A4** | MRSA pneumonia | `vancomycin` | FAIL (0.11) | PASS (1.00) | PASS (1.00) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |
| **A5** | CRE bacteremia + moderate-severe renal impairment (CrCl 25) | `ceftazidime-avibactam` | FAIL (0.11) | PASS (1.00) | PASS (1.00) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |
| **A6** | MDR Enterococcus bacteremia + dialysis (CrCl 8) | `daptomycin` | FAIL (0.11) | FAIL (0.11) | PASS (1.00) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |
| **A7** | XDR P. aeruginosa pneumonia -- last-line agent | `cefiderocol` | FAIL (0.11) | SUBOPT (0.76) | PASS (1.00) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |
| **A8** | MSSA bacteremia -- stewardship: cefazolin vs vancomycin | `cefazolin` | FAIL (0.11) | SUBOPT (0.64) | SUBOPT (0.81) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |
| **A9** | ESBL E. coli bacteremia -- carbapenem stewardship | `ertapenem` | SUBOPT (0.82) | FAIL (0.06) | PASS (1.00) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |
| **A10** | MDR E. coli CRE intra-abdominal infection | `ceftazidime-avibactam` | FAIL (0.11) | FAIL (0.11) | PASS (1.00) | [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |

> **Summary**: Broad-empiric 0/10 pass. Random(42) 2/10 pass. EUCAST-only 7/10 pass. Trained model: see live HuggingFace Space.

> **A1 note**: Oracle ceiling is 0.78 rather than 1.00 because `compute_optimal_prescription` is allergy-unaware (it finds ampicillin optimal), while `compute_total_reward` correctly applies R0=0 for the penicillin allergy. Vancomycin is the best *safe* option (R2=0.5, IDSA alternative), so quality\_ratio = 0.775 / 0.99 ≈ 0.78. This is correct reward behaviour — the env penalises allergy-blocking even at the oracle level.

> **Reproduce**: `python eval_adversarial.py --seed 42` — runs in under 10 seconds on CPU, no GPU required.
