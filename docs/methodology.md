# Methodology Notes

We model the rendered image as

\[ I = R(\theta; x) + \eta, \qquad \eta \sim \mathcal{N}(0, \Sigma) \]

with scene parameters \(\theta\) held fixed and pose \(x \in SE(3)\).
Linearising the renderer around a nominal pose yields

\[ R(\theta; \exp(\xi) x) \approx R(\theta; x) + J \xi, \]

where \(J = \partial R / \partial \xi\) is evaluated at \(\xi = 0\).
Under the Gaussian noise model, the Fisher information matrix is

\[ \mathcal{I}(x) = J^\top \Sigma^{-1} J. \]

The (unbiased) Cramér–Rao bound follows immediately:

\[ \operatorname{Cov}(\hat{\xi}) \succeq \mathcal{I}(x)^{-1}. \]

For multi-agent settings, each agent contributes information in its local
frame.  Transporting to a global frame uses the SE(3) adjoint
\(\mathrm{Ad}_g\):

\[ \widetilde{\mathcal{I}}_a = \mathrm{Ad}_{g_a}^{\top} \mathcal{I}_a \mathrm{Ad}_{g_a}. \]

Summing across agents provides the joint bound.  The greedy tile
selection routine evaluates objective functions such as `logdet`,
`trace`, or the minimum eigenvalue with an optional ridge to improve
numerical conditioning.
