# Method and implementation contract

## Boundary migration in fixed coordinates

After current-domain BN fitting, freeze BN and CNN14. The same features `h` are used for anchor and migrated classifiers:

```text
z0(h)   = W0 h + b0
zr(h)   = z0(h) + (alpha / max_rank) B_r A_r h + delta_b_r
z(h,rho)= z0(h) + rho (zr(h) - z0(h))
```

The source classifier remains the anchor at every arrival. Pairwise row differences set equal-score plane normals; bias differences set intercepts. Only winning portions of these planes form multiclass decision boundaries. Removing the mean class row preserves class-score differences. Thus identifiable weight-residual rank is at most `C-1`.

The configs enumerate all integer ranks `1..min(9,C-1)`, plus rank 0 as the anchor. The training factor scale is always `8/3`: `alpha=24,max_rank=9` for DCASE/TAU and `alpha=8,max_rank=3` for ADIL. These are factorization conventions, not different deployed budgets.

## Training versus final selection

`train_cached_hcrg.py` is retained for trajectory parity. The public wrapper explicitly sets `dual_learning_rate=0`; its older standalone default is not the paper recipe. Class-balanced CE trains the factors and bias. Rank growth uses a centered-gradient leading singular pair. At each epoch a provisional 101-point radial search selects a feasible validation checkpoint; its restored state initializes the next rank.

The checkpoint also saves the **unscaled winning state** at each rank as `candidate_raw_states`. Final selection operates on those frozen paths. It does not reuse the trainer's provisional `selected_state_dict`, optimize held-out labels, or retrain a path for each budget.

## NFR and corrections

For each class present in a split, divide negative flips by **all examples of that class**, then average across present classes. A negative flip is correct under the source-head anchor and incorrect after migration. Corrections reverse that condition. This yields the identity

```text
macro_accuracy_gain = correction_rate - negative_flip_rate
```

Config budgets are fractions: fit `0.005`, validation `0.010`. Reported metrics are percentages, so these are `0.5%` and `1.0%`. The rank tolerance is `0.5` percentage points. These are empirical deployment choices, not population-risk guarantees.

## Event enumeration and joint rank-path choice

Along an affine path, all true-versus-rival margins are affine functions of `rho`. Their positive intersection is at most one interval per sample, giving at most an entering and an exiting event. Sorting fit/validation events yields intervals where correctness, accuracy and NFR are constant. Anchor-correct samples can only exit in the absence of ties, giving nondecreasing NFR on a path.

The selector considers intervals across all frozen ranks. Among feasible states it finds the best validation accuracy, chooses the smallest rank within 0.5 pp, then orders by accuracy, NFR and smaller `rho`. Rank 0 may win. The retained research score also contains net correction rate, redundant with accuracy when the anchor is fixed.

`events.py` retains the research implementation's slope tolerance (`1e-12`), root merging (`1e-11`), anchor representative at zero and midpoints elsewhere. Coverage excludes isolated roots and persistent zero-margin ties. This is not an exact-real-arithmetic solver over all possible tie states. The public exporter directly recomputes the chosen predictions and refuses export if event metrics or observed budgets disagree. It does not silently change the selector to rescue a failed numerical case.

## Deployment and retention

`head.npz` stores `delta_weight` and `delta_bias` **after multiplying by rho**. `SAFBMHead.from_npz` merges them into the source linear layer. There is no test-time rank/path search. Float64 is the deployment audit default; casting to float32 can change predictions arbitrarily near ties and should be rechecked for the intended deployment.

A sequence retains one BN/head pair per domain. The supplied evaluator sends each example to the seen branch with minimum softmax entropy, with first-branch tie breaking. It recomputes routing separately for anchor and SAF-BM. Even unchanged branches can lose historical accuracy as the candidate set grows; boundary-preserving logit scaling can also change entropy. Local feasibility therefore says nothing by itself about historical retention.
