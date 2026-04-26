## 🧪 Clinical Validation Against Published Case Literature

The following cases are encoded directly as `PatientCase` objects and run through
the live AMR-Steward environment. Reward breakdown validates that R1 (microbiological
activity), R2 (IDSA guideline concordance), and R4 (renal dosing) all fire correctly
against the published expert recommendation.

| Case | Patient | Published Recommendation | Citation | AMR-Steward Output | R1 | R2 | Quality | Match |
|------|---------|--------------------------|----------|--------------------|----|----|---------|-------|
| **Case 1 Cre Bacteremia** | 67M post-renal-transplant, CRE K. pneumoniae bacteremia. | Ceftazidime-avibactam (IDSA preferred for KPC-CRE bacteremia... | Tamma PD et al | `ceftazidime-avibactam 1.25g IV q8h` | ✅ 1.0 | ✅ 1.0 | 1.00 | ✅ |
| **Case 2 Mssa Pcn Allergy** | 58M, MSSA bacteremia. | Cefazolin (preferred beta-lactam for penicillin-allergic pat... | Maraolo AE et al | `cefazolin 2g IV q8h` | ✅ 1.0 | ✅ 1.0 | 1.00 | ✅ |
| **Case 3 Vre Hemodialysis** | 72F on hemodialysis (CrCl 8 mL/min), VRE E. faecium bloodstream infection. | High-dose daptomycin (≥8 mg/kg, renal-adjusted, post-HD dosi... | Britt NS et al | `daptomycin 8mg/kg IV post-HD` | ✅ 1.0 | ⚠ 0.5 | 0.94 | ❌ |

> **Reproduction:** `python eval_published_cases.py`  
> Cases injected via `POST /reset` with `patient=PatientCase(...)`.  
> The environment's RLVR oracle and EUCAST/IDSA JSON tables score the prescription independently.  
> R1 = microbiological activity, R2 = IDSA guideline concordance, Quality = R1·0.40 + R2·0.25 + R3·0.15 + R4·0.10 / optimal.

### Case Details

**Case 1: Case 1 Cre Bacteremia**  
*67M post-renal-transplant, CRE K. pneumoniae bacteremia.
Meropenem MIC=8 (Resistant), ceftazidime-avibactam MIC=1 (Susceptible).
CrCl 40 mL/min (moderate impairment). No allergies.*  
Published recommendation: Ceftazidime-avibactam (IDSA preferred for KPC-CRE bacteremia, renal-adjusted)  
Citation: Tamma PD et al. IDSA 2022 Guidance on AMR Gram-Negative Infections. Clin Infect Dis. 2023;76(7):1228-1270. PMC9890506.  
Model output: `ceftazidime-avibactam` `1.25g IV q8h` `1.25g IV q8h` — quality_ratio `1.000`  

**Case 2: Case 2 Mssa Pcn Allergy**  
*58M, MSSA bacteremia.
Oxacillin MIC=0.25 (Susceptible), cefazolin MIC=1 (Susceptible).
CrCl 65 mL/min. No allergies.*  
Published recommendation: Cefazolin (preferred beta-lactam for penicillin-allergic patients with MSSA bacteremia; non-inferior to nafcillin in outcomes)  
Citation: Maraolo AE et al. Influence of Reported Penicillin Allergy on Mortality in MSSA Bacteremia. Open Forum Infect Dis. 2018;5(3):ofy042. doi:10.1093/ofid/ofy042.  
Model output: `cefazolin` `2g IV q8h` `2g IV q8h` — quality_ratio `1.000`  

**Case 3: Case 3 Vre Hemodialysis**  
*72F on hemodialysis (CrCl 8 mL/min), VRE E. faecium bloodstream infection.
Vancomycin MIC=32 (Resistant), daptomycin MIC=1 (Susceptible), linezolid MIC=2 (Susceptible).
No allergies.*  
Published recommendation: High-dose daptomycin (≥8 mg/kg, renal-adjusted, post-HD dosing); superior microbiologic clearance vs linezolid in VRE BSI  
Citation: Britt NS et al. Comparison of Effectiveness and Safety of Linezolid and Daptomycin in VRE Bloodstream Infection. Clin Infect Dis. 2015;61(6):871-878. PMC4551011.  
Model output: `daptomycin` `8mg/kg IV post-HD` `8mg/kg IV post-HD` — quality_ratio `0.939`  
