# MEETING_GUIDED_NOTES_HINGLISH.md
# Supervisor Meeting — Guided Notes (English/Hinglish)
# For: Mohd Zamin Quadri | Meeting with Dominik Fuchsgruber + Elena Natterer
# Last verified: 2026-03-14

---

## Format: Question → What They're Actually Asking → How to Answer

Each section has:
- The question (as they might phrase it)
- What they really want to know
- A script you can say (English, with Hinglish reminders in [brackets])
- What NOT to say

---

## PART 1: Your Core Contribution

---

### Q1: "What is the main contribution of your thesis?"

**They want to know:** Is this just applying existing methods, or did you identify a real gap and fill it?

**Say this:**
"The main gap I identified is that agent-based transport simulators like MATSim are too slow for policy analysis at scale — a single run takes hours. So I trained a GNN surrogate to approximate the simulator's output. The contribution is then: (1) showing that this GNN can predict policy-induced traffic changes reasonably well, and (2) adding uncertainty quantification so planners know when to trust the prediction."

**[Hinglish reminder]:** "Contribution do parts mein hai — ek GNN surrogate training, doosra uncertainty quantification. Dono ko clearly explain karo."

**Do NOT say:** "I implemented MC Dropout." — That's a method, not a contribution.

---

### Q2: "Why GNN specifically? Why not a simpler model?"

**They want to know:** Did you make a principled choice, or did you just use what was available?

**Say this:**
"Road networks are graphs by nature — nodes are road segments, edges are shared junctions. Spatial relationships matter: if one road is congested, neighboring roads are affected. GNNs can propagate this information. A feed-forward network would miss this structure entirely."

**[Hinglish reminder]:** "Road network = graph. GNN = structure-aware. Simple model = structure-blind. Yahi answer hai."

**Do NOT say:** "Elena suggested GNN." — Even if true, own the decision.

---

### Q3: "What is the line graph transformation and why did you need it?"

**They want to know:** Do you understand why your input representation is the way it is?

**Say this:**
"PyTorch Geometric predicts node-level features. But road volume is an edge property — each road segment is an edge in the original network. The line graph transform converts edges to nodes, so road segments become nodes and shared intersections become edges. This lets us use standard node-level GNN layers."

**[Hinglish reminder]:** "Line graph = edges ko nodes bana do. Intersection = new edges. Isliye 31,635 nodes per graph."

---

## PART 2: Model Performance

---

### Q4: "How well does the model perform?"

**They want to know:** Is R²=0.59 acceptable? Can you put it in context?

**Say this:**
"Trial 8 achieves R²=0.5957, MAE=3.96 veh/h, RMSE=7.12 veh/h on the test set. For context, the target variable is the change in car volume — not absolute volume. This is a much harder prediction problem because change signals are smaller and noisier than absolute values. The model explains about 60% of variance in policy-induced traffic changes, which I argue is a useful starting point for exploratory policy analysis."

**[Hinglish reminder]:** "R²=0.59 hai, lekin target = change in volume, not absolute. Change predict karna harder hai. Yahi context dena hai."

**Do NOT say:** "R² is quite good." — Dominik will push back. Acknowledge it is moderate.

---

### Q5: "Why did you use 1,000 scenarios out of 10,000?"

**They want to know:** Is the choice principled or was it just computational limitation?

**Say this:**
"Computational cost was the primary constraint — each MATSim run is expensive to store and process. 1,000 scenarios gives 800 training graphs, which is sufficient to fit this architecture. I treat using more data as a clear avenue for future improvement."

**[Hinglish reminder]:** "1,000 of 10,000 isliye liye kyunki compute/storage constraint tha. Acknowledge karo ki more data = better."

---

### Q6: "What was the effect of your hyperparameter ablations?"

**They want to know:** Did you do a systematic search or random trials?

**Say this:**
"The trials were not a full grid search — they were sequential, each motivated by the previous result. Key findings: weighted loss (T3/T4) hurt performance, likely because outlier upweighting confused the model. Reducing batch size from 32 to 8 helped, as did reducing dropout from 0.3 to 0.2 in T8. The 80/10/10 split gave more test graphs for a more reliable evaluation. I acknowledge a proper Bayesian hyperparameter search would be stronger."

**[Hinglish reminder]:** "Sequential trials the, grid search nahi. Acknowledge kar ki full search nahi kiya."

---

## PART 3: Uncertainty Quantification

---

### Q7: "How did you evaluate uncertainty quality?"

**They want to know:** Are your uncertainty estimates meaningful, or are they just noise?

**Say this:**
"I used Spearman rank correlation between predicted uncertainty (MC Dropout variance) and actual absolute error. For T8, ρ=0.4820 on 3.16 million test nodes. This means the model's uncertainty is positively correlated with where it makes errors — it knows when it doesn't know. For comparison, ensemble methods gave ρ≈0.10–0.16, so MC Dropout is significantly stronger here."

**[Hinglish reminder]:** "Spearman ρ = uncertainty aur actual error ka correlation. T8 mein ρ=0.48, ensemble mein 0.10-0.16. MC Dropout much better."

**Do NOT say:** "The uncertainty is well-calibrated." — Calibration means something specific (reliability diagrams, ECE). Don't claim calibration unless verified.

---

### Q8: "What is conformal prediction and why did you use it?"

**They want to know:** Can you explain the guarantee, not just the name?

**Say this:**
"Conformal prediction gives a distribution-free coverage guarantee. Given a calibration set, I find the quantile q such that the model's prediction intervals cover the true value at exactly the desired rate. For T8 at 95%: q=14.68 veh/h, achieved coverage=95.01% on 50 evaluation graphs — the guarantee holds empirically. This is useful for planners because it's a hard guarantee, not an approximation."

**[Hinglish reminder]:** "Conformal = coverage guarantee. 95% matlab 95% test nodes ka true value interval ke andar hai. q=14.68 veh/h."

---

### Q9: "Why is MC Dropout a valid Bayesian approximation?"

**They want to know:** Did you think about this critically, or did you just use a popular technique?

**Say this:**
"Gal & Ghahramani (2016) showed that a neural network with dropout applied at test time is equivalent to a variational Bayesian approximation over the posterior of the weights. Each forward pass with random dropout samples a different sub-network, and the variance across T=30 passes approximates posterior predictive variance. I acknowledge this is a loose approximation — the variational family is limited and the prior is implicit — but it is computationally tractable and empirically useful here, as the ρ=0.48 shows."

**[Hinglish reminder]:** "MC Dropout = approximate Bayes. Gal & Ghahramani 2016 citation. Acknowledge limitations."

---

### Q10: "Why does ensemble give worse UQ than MC Dropout here?"

**They want to know:** Do you have a hypothesis or is this surprising to you?

**Say this:**
"This is somewhat surprising and I have two hypotheses. First, the ensemble in Experiment A used only 5 runs of the same model — the diversity between ensemble members may be too low since they share the same architecture and similar training trajectories. Second, we used models T7 and T8 for the ensemble, but these are very similar architecturally — a proper ensemble would use diverse architectures. I treat this as a finding: MC Dropout is more computationally efficient and performs better here, but understanding why would require more experiments."

**[Hinglish reminder]:** "Ensemble ka diversity kam tha — same architecture, similar training. Isliye MC Dropout better. Honest answer hai."

---

## PART 4: Dominik's Likely Hard Questions

---

### Q11: "Have you considered OOD detection — can your model identify when a policy scenario is out-of-distribution?"

**They want to know:** Are you aware of the limitations of applying surrogate predictions to novel scenarios?

**Say this:**
"I have not implemented explicit OOD detection in this work. The UQ methods I use — MC Dropout variance and conformal prediction — are both trained and calibrated on in-distribution data. They can signal high uncertainty for unusual inputs, but this is not a formal OOD detector. This is a real limitation: if a policy scenario is genuinely novel (e.g., a new district that does not exist in training data), the model may produce confident wrong predictions. I flag this as future work."

**[Hinglish reminder]:** "OOD detection nahi kiya. Acknowledge karo. Future work mein mention karo."

---

### Q12: "Why did you use Spearman rank correlation as the UQ metric?"

**They want to know:** Is this the right metric, or was it a convenient choice?

**Say this:**
"Spearman rank correlation measures whether high uncertainty predictions correspond to high actual errors, independent of the scale of uncertainty. It is the standard metric for uncertainty-error alignment in regression UQ literature. An alternative would be PICP (prediction interval coverage probability) which I also computed via conformal prediction. I did not use Expected Calibration Error because that requires binning and is more appropriate for classification tasks."

**[Hinglish reminder]:** "Spearman ρ = standard UQ metric for regression. PICP bhi compute kiya (conformal). ECE = classification ke liye better hai."

---

### Q13: "Is your uncertainty estimate calibrated?"

**They want to know:** Are ±1σ intervals really 68% coverage?

**Say this:**
"Not in the frequentist sense. The raw MC Dropout sigma is NOT a calibrated standard deviation — k₉₅ = 11.65 empirically, whereas a calibrated normal would give k₉₅ = 1.96. The conformal prediction intervals ARE calibrated by construction — they achieve exact empirical coverage. For the MC Dropout sigma itself, I do not claim calibration and avoid making probability statements about it."

**[Hinglish reminder]:** "MC Dropout sigma calibrated nahi hai (k95=11.65 vs 1.96). Conformal IS calibrated. Dono ko clearly separate karo."

**This is critical — Dominik WILL ask this.**

---

### Q14: "What would you do differently if you started again?"

**They want to know:** Your critical self-assessment.

**Say this:**
"Three things: (1) Use a proper hyperparameter search (Bayesian optimization) instead of sequential trials. (2) Use more of the available data — training on all 10,000 scenarios. (3) Explore heteroscedastic models that learn input-dependent uncertainty directly, rather than using MC Dropout which treats uncertainty as a post-hoc approximation."

**[Hinglish reminder]:** "3 things: proper HP search, more data, heteroscedastic model. Yahi critical answer hai."

---

## PART 5: Quick Reference Numbers (carry this to meeting)

```
Best model:      Trial 8
R²:              0.5957
MAE:             3.96 veh/h
RMSE:            7.12 veh/h
MC Dropout ρ:    0.4820  (T8, 30 samples, 100 graphs)
Ensemble ρ:      0.1035–0.1601  (much worse)
Conformal 90%:   q=9.92 veh/h, coverage=90.02%
Conformal 95%:   q=14.68 veh/h, coverage=95.01%
k95 (raw σ):     11.65  (NOT 1.96 — sigma is not calibrated)
Selective pred:  Reject top 50% uncertain → MAE drops 39.9%
Narrow intervals: 65.8% narrower for low-uncertainty predictions
Test nodes:      3,163,500  (100 graphs × 31,635 nodes)
MC Dropout time: 228.25 min  (100 graphs × 30 samples)
```

---

## PART 6: What to Say If You Don't Know

It is better to say:
> "That's a good point — I haven't explored that in this thesis, but I think [X] would be the right approach. I'd be happy to discuss it."

Than to guess and be wrong. Dominik respects intellectual honesty.

**[Hinglish reminder]:** "Agar answer nahi pata — honestly bol do. 'I haven't looked at this but I think X.' Guess mat karo."

---

## PART 7: Opening Statement (first 2 minutes)

Prepare this short statement to open the meeting confidently:

> "My thesis addresses a practical bottleneck: MATSim simulations take hours per scenario, making real-time policy exploration impossible. I trained a GNN surrogate on 1,000 Paris simulation scenarios to predict policy-induced traffic changes at the road-segment level. The best model achieves R²=0.596 and MAE=3.96 veh/h. I then added two layers of uncertainty quantification: MC Dropout for uncertainty ranking (Spearman ρ=0.482) and conformal prediction for coverage-guaranteed intervals (95% coverage, width ±14.7 veh/h). Together, these allow a planner to query the surrogate and receive not just a prediction but a trust signal — which is essential when the output informs real transport policy."

**[Hinglish reminder]:** "Yeh opening statement memorize karo. 2 minutes mein clear picture deta hai — problem, method, result, implication."
