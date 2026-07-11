# Mathematical Formulation

## AC power flow

For bus `i`, the active and reactive power-balance equations are

\[
P_{Gi}-P_{Di}=V_i\sum_jV_j\left[G_{ij}\cos(\theta_i-\theta_j)+B_{ij}\sin(\theta_i-\theta_j)\right],
\]

\[
Q_{Gi}-Q_{Di}=V_i\sum_jV_j\left[G_{ij}\sin(\theta_i-\theta_j)-B_{ij}\cos(\theta_i-\theta_j)\right].
\]

The internal solver uses a full Newton-Raphson formulation with slack, PV, and PQ buses,
off-nominal complex transformer taps, bus shunts, branch charging, and aggregate generator
reactive-power limits. When a PV bus requires reactive generation outside the aggregate limits of
its online generators, it is converted to PQ at the corresponding limit and the equations are
resolved.

## ORPD decision vector

Every optimizer searches a normalized vector

\[
\mathbf z\in[0,1]^D.
\]

One common decoder maps it to

\[
\mathbf u=[V_G,\,T,\,Q_C],
\]

where generator voltages may be continuous and transformer/shunt settings may be decoded to
explicit engineering steps.

## Active-power loss

\[
P_{loss}=\sum_{(i,j)}\Re\{S_{ij}+S_{ji}\}.
\]

## Voltage deviation

\[
VD=\sum_{i\in\mathcal N_{PQ}}|V_i-1|.
\]

## Kessel-Glavitsch L-index

Partition the bus-admittance matrix into load and generator/reference blocks. Define

\[
\mathbf F=-\mathbf Y_{LL}^{-1}\mathbf Y_{LG}.
\]

For load bus `j`,

\[
L_j=\left|1-\frac{\sum_iF_{ji}V_i}{V_j}\right|,
\qquad L_{max}=\max_jL_j.
\]

Branch thermal loading is maintained as a separate engineering metric and is never interpreted
as the L-index.

## Multi-objective aggregation

The software can evaluate

\[
F=w_1\frac{P_{loss}}{s_P}+w_2\frac{VD}{s_V}+w_3\frac{L_{max}}{s_L},
\]

while retaining every raw component independently.

## Constraints

The evaluator audits bus-voltage limits, aggregate generator active and reactive limits,
transformer and shunt device validity, branch thermal limits, and power-flow convergence.
Comparisons use feasibility-first ordering by default:

1. a feasible candidate dominates an infeasible candidate;
2. among feasible candidates, lower objective is preferred;
3. among infeasible candidates, lower normalized total violation is preferred.

## Robust scenario aggregation

For scenario objective values `F_s` and explicit weights `w_s`, supported measures are

\[
E[F]=\sum_sw_sF_s,
\]

\[
E[F]+\lambda\,\mathrm{Std}[F],
\]

\[
\max_sF_s,
\]

and empirical weighted CVaR at a user-selected confidence level.
