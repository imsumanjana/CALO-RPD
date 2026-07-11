# CALO Methodology

## Cognitive Adaptive Learning Optimizer

CALO is a search-state-aware optimization architecture. It does not reduce to TLBO with a changed
random-number distribution. At each iteration, it measures the state of the search, queries a compact
AI policy controller, executes one of six bounded learning modes, records successful moves, and
updates diversity and stagnation information.

## Cognitive state

The normalized state contains:

- population diversity;
- recent best-objective improvement;
- recent median-population improvement;
- normalized stagnation duration;
- feasible-solution ratio;
- normalized mean constraint violation;
- elite spread;
- remaining evaluation budget;
- six operator-success features.

The release policy therefore uses a 14-dimensional input vector.

## AI policy controller

The controller is a compact PyTorch actor-critic network with two fully connected hidden layers.
The policy head produces probabilities for six learning modes. A second head produces bounded
continuous controls for exploitation, peer learning, exploration variance, memory contribution,
recovery intensity, and recovery population fraction. A value head supports reproducible
actor-critic/PPO-style training.

The AI controller is used only by CALO. The nineteen primary baseline algorithms receive no AI
assistance.

## Learning modes

### 1. Teacher-guided exploitation

\[
X_i'=X_i+\alpha_t|Z_1|(X_{best}-X_i)+\beta_tZ_2(X_{best}-\bar X),
\qquad Z_1,Z_2\sim\mathcal N(0,1).
\]

The first term preserves direction toward the current best. The second provides controlled adaptive
variation.

### 2. Contrastive peer learning

\[
X_i'=X_i+\gamma_t r_1(X_{better}-X_i)+\delta_t r_2(X_i-X_{diverse}).
\]

### 3. Self-reflective memory learning

\[
X_i'=X_i+\eta_t r_1(P_i-X_i)+\mu_tM_{success}.
\]

### 4. Adaptive exploration

\[
X_i'=X_{reference}+\sigma_tZ.
\]

Because the optimization space is normalized, the perturbation scale is independent of engineering
units.

### 5. Feasibility recovery

When the feasible ratio is poor, moves are biased toward a feasible elite when one exists and toward
the lowest-violation region otherwise. Candidate feasibility is still determined only by the common
physical evaluator.

### 6. Stagnation escape

When the configured stagnation window is reached, a controlled fraction of poor learners is perturbed
around the elite region while leading solutions are preserved.

## Success memory

A bounded deque stores accepted directions, operator identity, step norm, normalized objective gain,
and feasibility gain. Recency weighting prevents unbounded history and allows the population to use
recently successful movement information.

## Reward

The training reward combines normalized objective improvement, feasible-ratio improvement, useful
diversity recovery, constraint-violation penalty, and computational-overhead penalty. Individual
components are maintained separately in the methodology and training code.

## Training/test separation

The packaged policy is trained only on documented synthetic numerical function families. Its metadata
explicitly states whether final power-system test cases were used for training. Final comparative runs
use a frozen checkpoint by default. Any different policy or online adaptation is a distinct experiment
configuration and is preserved in provenance.

## Ablation suite

The application provides:

1. classical TLBO;
2. legacy Gaussian MTLBO;
3. CALO without AI;
4. CALO without success memory;
5. CALO without stagnation recovery;
6. CALO without diversity feedback;
7. complete CALO.
