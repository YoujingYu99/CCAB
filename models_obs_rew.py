import torch
import torch.distributions as D
import pyro.distributions as pyroD


class SuffStats:
    """
    Class that maintains the sufficient statistics for a distribution.
    This is a base class that can be extended for specific distributions.
    """

    def __init__(self):
        pass

    def update(self, x, confidence=1.0):
        """
        Update sufficient statistics with a new observation x.
        confidence: weight of the new observation (default 1.0)
        """
        raise NotImplementedError("update method must be implemented by subclass")

    def to_state(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict):
        raise NotImplementedError


class ConjugateModel:
    """
    Base class for a conjugate model. This should be extended for specific likelihood-prior pairs.
    """

    # subclasses MUST set this to specific sufficient statistics class, e.g. SuffStatsGaussian
    SUFFSTATS_CLS = None  # type: type[SuffStats]

    def __init__(self):
        pass

    def update(self, x, confidence=1.0):
        """
        Add a new observation and update posterior parameters.
        """
        raise NotImplementedError("update method must be implemented by subclass")

    def post_params(self) -> dict:
        """
        Return current posterior parameters as a dictionary.
        """
        raise NotImplementedError("post_params method must be implemented by subclass")

    def sample_post_dist(self):
        """
        Sample from the posterior distribution over parameters.
        """
        raise NotImplementedError("sample_post_dist method must be implemented by subclass")

    def _pred_dist_params(self):
        """
        Return parameters of the posterior predictive distribution.
        """
        raise NotImplementedError("_pred_dist_params method must be implemented by subclass")

    def pred_lh(self, x):
        """
        Return the predictive likelihood of a new observation x.
        """
        raise NotImplementedError("pred_lh method must be implemented by subclass")


    # Extra methods for serialization and reconstruction of the model state during resampling of particles in a particle filter.
    def to_state(self) -> dict:
        """
        For all conjugate models, the minimal state representation needed to reconstruct the object is \\
        just the sufficient statistics, since the posterior can be fully reconstructed from the sufficient \\
        statistics and the prior parameters (which are fixed and known).
        """
        return {"suffstats": self.suffstats.to_state()}

    @classmethod
    def from_state(cls, hyp_params: dict, state: dict):
        """
        Reconstruct a conjugate model object from its minimal state representation, \\
        which is just the sufficient statistics and the prior parameters.
        """
        # check that subclass has set SUFFSTATS_CLS
        if cls.SUFFSTATS_CLS is None:
            raise TypeError(f"{cls.__name__} must set SUFFSTATS_CLS")

        # Re-init the object with the prior parameters
        obj = cls(**hyp_params)
        # Then reconstruct the sufficient statistics from the state
        obj.suffstats = cls.SUFFSTATS_CLS.from_state(state["suffstats"])
        # Finally, update the posterior parameters based on the sufficient statistics
        obj._update_posterior()

        return obj






class SuffStatsGaussian(SuffStats):
    """
    Class that maintains the sufficient statistics for a (multivariate) Gaussian distribution.
    Sufficient statistics are:
        n: number of observations
        sum_x: sum of observations
        sum_xx: sum of outer products of observations
    """

    def __init__(self, d: int):
        self.n      = 0
        self.sum_x  = torch.zeros(d)
        self.sum_xx = torch.zeros(d, d)

    def update(self, x: torch.Tensor, confidence: float = 1.0):
        """
        Update sufficient statistics with a new observation x.
        x: tensor of shape [d]
        """
        self.n += confidence
        self.sum_x += confidence * x
        self.sum_xx += confidence * torch.outer(x, x)

    @property
    def mean(self):
        """
        Mean of the observations.
        """
        if self.n == 0:
            return None
        return self.sum_x / self.n

    @property
    def scatter(self):
        """
        S = Σ_i (x_i - x̄)(x_i - x̄)^T = Σ_i (x_i)(x_i)^T - n * (x̄)(x̄)^T

            Scatter around the empirical mean.
        """
        if self.n == 0:
            return None
        mu = self.mean
        return self.sum_xx - self.n * torch.outer(mu, mu)

    def to_state(self):
        """
        Minimal state representation needed to reconstruct the sufficient statistics.
        """
        return {
            "n":      float(self.n),
            "sum_x":  self.sum_x,
            "sum_xx": self.sum_xx,
        }

    @classmethod
    def from_state(cls, state):
        """
        Reconstruct sufficient statistics from its minimal state representation.
        """
        d = int(state["sum_x"].numel())
        obj = cls(d)
        obj.n = float(state["n"])
        obj.sum_x = state["sum_x"].clone()
        obj.sum_xx = state["sum_xx"].clone()
        return obj


class ConjugateGaussianInvWish(ConjugateModel):
    SUFFSTATS_CLS = SuffStatsGaussian

    def __init__(self, mu0, kappa0, nu0, Lambda0):
        """
        Conjugate Gaussian-Inverse Wishart distribution for a posterior over a multivariate Gaussian with unknown mean & covariance.

        mu0:        prior mean mean [d]
        kappa0:     pseudocount of prior measurements
        nu0:        degrees of freedom > d-1
        Lambda0:    scale matrix [d,d], positive definite
        """
        assert type(mu0) == torch.Tensor, "prior mean mu0 must be a torch tensor"
        assert mu0.dim() == 1, "prior mean mu0 must be a 1D tensor"
        self.mu0 = mu0

        assert (type(kappa0) == float), "pseudocount of prior measurements kappa0 must be a float"
        assert kappa0 > 0, "pseudocount of prior measurements kappa0 must be > 0"
        self.kappa0 = float(kappa0)

        assert type(nu0) == float, "nu0 must be a float"
        assert (nu0 > mu0.shape[0] - 1), f"degrees of freedom nu0 must be > {mu0.shape[0]-1}"
        self.nu0 = float(nu0)

        assert (type(Lambda0) == torch.Tensor), "scale matrix Lambda0 must be a torch tensor"
        assert Lambda0.dim() == 2, "scale matrix Lambda0 must be a 2D tensor"
        assert (Lambda0.shape[0] == Lambda0.shape[1] == mu0.shape[0]), "scale matrix Lambda0 must be of shape (d,d)"
        # check positive definiteness
        eigvals = torch.linalg.eigvalsh(Lambda0)
        assert torch.all(eigvals > 0), "scale matrix Lambda0 must be positive definite"
        self.Lambda0 = Lambda0

        self.d = mu0.shape[0]

        # Maintain sufficient statistics
        self.suffstats = SuffStatsGaussian(self.d)

        # Posterior params start at prior
        self.reset_posterior()

    def reset_posterior(self):
        self.mu_n = self.mu0.clone()
        self.kappa_n = self.kappa0
        self.nu_n = self.nu0
        self.Lambda_n = self.Lambda0.clone()

    def update(self, x: torch.Tensor, confidence: float = 1.0):
        """
        Add a new observation and update posterior parameters.
        """
        self.suffstats.update(x, confidence=confidence)
        self._update_posterior()

    def _update_posterior(self):
        """
        Update posterior parameters based on current sufficient statistics.
        """
        if self.suffstats.n == 0: # do not update posterior if there are no datapoints
            return

        x_bar = self.suffstats.mean
        n = self.suffstats.n
        S = self.suffstats.scatter

        # Update posterior of kappa
        """
        κ_n = κ_0 + n
        """
        self.kappa_n = self.kappa0 + n

        # Update posterior of mean
        """
        μ_n = ( (κ_0 / (κ_n)) * μ0 ) + ( (n / (κ_n)) * x̄ )
        """
        self.mu_n = ((self.kappa0 / self.kappa_n) * self.mu0) + ((n / self.kappa_n) * x_bar)


        # Update posterior of nu
        """
        ν_n = ν_0 + n
        """
        self.nu_n = self.nu0 + n

        # Update posterior of Lambda
        """
        Λ_n = Λ_0 + S + ((κ_0 * n )/ (κ_n)) * (x̄ - μ0)(x̄ - μ0)^T
        """
        diff = (x_bar - self.mu0).unsqueeze(1)  # column vector, difference between sample mean and prior mean
        self.Lambda_n = (self.Lambda0 + S + ((self.kappa0 * n) / (self.kappa_n)) * (diff @ diff.T))

    def post_params(self):
        return {
            "mu_n":     self.mu_n,
            "kappa_n":  self.kappa_n,
            "nu_n":     self.nu_n,
            "Lambda_n": self.Lambda_n,
        }

    def sample_post_dist(self):
        """
        Sample from the posterior distribution over the mean vector and covariance matrix.
        """
        # Sample covariance matrix
        """
        To get: Σ ~ IW(ν_n, Λ_n)    
        Ω ~ W(ν_n, Λ_n^{-1})        we can sample a precision matrix from the Wishart distribution,
        Σ = Ω^{-1}                  then invert
        """
        Prec  = D.Wishart(df=self.nu_n, covariance_matrix=torch.linalg.inv(self.Lambda_n)).sample()
        Sigma = torch.linalg.inv(Prec)
        # Sample mean from Gaussian
        """
        μ ~ 𝒩(μ_n, Σ / κ_n)
        """
        mu = D.MultivariateNormal(loc=self.mu_n, covariance_matrix=(Sigma / self.kappa_n)).sample()

        return mu, Sigma

    def _pred_dist_params(self):
        """
        Returns parameters of the multivariate Student-t predictive distribution (uncerteanty over mean and variance is integrated out).
        """
        df = self.nu_n - self.d + 1
        scale = (self.Lambda_n * (self.kappa_n + 1)) / (self.kappa_n * df)

        return {
            "df":    df,
            "loc":   self.mu_n,
            "scale": scale,
        }

    def pred_lh(self, x: torch.Tensor):
        """
        Returns the predictive likelihood of a new observation x.
        """
        params = self._pred_dist_params()
        pred_dist = pyroD.MultivariateStudentT(df=params["df"], loc=params["loc"], scale_tril=torch.linalg.cholesky(params["scale"]),)
        return torch.exp(pred_dist.log_prob(x))


class ConjugateGaussianInvGam(ConjugateModel):
    """
    Normal-Inverse-Gamma conjugate model for a spherical multivariate
    Gaussian:

        x | mu, sigma^2 ~ N(mu, sigma^2 I_d)

        sigma^2 ~ InvGamma(alpha_0, beta_0)

        mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0 I_d)

    The same scalar sigma^2 is shared across all dimensions.

    Posterior:

        sigma^2 | D ~ InvGamma(alpha_n, beta_n)

        mu | sigma^2, D
            ~ N(mu_n, sigma^2 / kappa_n I_d)

    where

        kappa_n = kappa_0 + n

        mu_n =
            (kappa_0 mu_0 + n x_bar) / kappa_n

        alpha_n =
            alpha_0 + n d / 2

        beta_n =
            beta_0
            + 1/2 SSE
            + (kappa_0 n)/(2 kappa_n)
              ||x_bar - mu_0||^2
    """

    SUFFSTATS_CLS = SuffStatsGaussian

    def __init__(self, mu0: torch.Tensor, kappa0: float, alpha0: float, beta0: float):
        """
        Conjugate Normal-Inverse-Gamma model for a spherical multivariate Gaussian.
            mu0:        prior mean, shape [d].
            kappa0:     Prior strength / pseudocount for the mean.
            alpha0:     Shape parameter of the Inverse-Gamma prior.
            beta0:      Scale parameter of the Inverse-Gamma prior.
        """

        assert type(mu0) == torch.Tensor, "mu0 must be a torch tensor"
        assert mu0.dim() == 1, "mu0 must be a 1D tensor"
        self.mu0 = mu0.clone()

        assert type(kappa0) == float, "kappa0 must be a float"
        assert kappa0 > 0, "kappa0 must be > 0"
        self.kappa0 = float(kappa0)

        assert type(alpha0) == float, "alpha0 must be a float"
        assert alpha0 > 0, "alpha0 must be > 0"
        self.alpha0 = float(alpha0)

        assert type(beta0) == float, "beta0 must be a float"
        assert beta0 > 0, "beta0 must be > 0"
        self.beta0 = float(beta0)
        
        self.d = mu0.shape[0]

        # Maintain sufficient statistics.
        self.suffstats = SuffStatsGaussian(self.d)

        # Initialize posterior to prior.
        self.reset_posterior()

    def reset_posterior(self):
        self.mu_n = self.mu0.clone()
        self.kappa_n = self.kappa0
        self.alpha_n = self.alpha0
        self.beta_n = self.beta0

    def update(self, x: torch.Tensor, confidence: float = 1.0):
        """
        Add an observation and update posterior parameters.
        """
        self.suffstats.update(x, confidence=confidence)
        self._update_posterior()

    def _update_posterior(self):
        """
        Update posterior parameters based on current sufficient statistics.

        Posterior:

            sigma^2 | D ~ InvGamma(alpha_n, beta_n)

            mu | sigma^2, D
                ~ N(mu_n, sigma^2 / kappa_n I_d)
        """
        if self.suffstats.n == 0: # do not update posterior if there are no datapoints
            return

        x_bar = self.suffstats.mean
        n = self.suffstats.n
        S = self.suffstats.scatter
        sse = torch.trace(S)  # sum of squared errors

        # Update posterior of kappa
        """
        κ_n = κ_0 + n
        """
        self.kappa_n = self.kappa0 + n

        # Update posterior of mean
        """
        μ_n = (κ_0 * μ0 + n * x̄) / κ_n
        """
        self.mu_n = ((self.kappa0 * self.mu0) + (n * x_bar)) / self.kappa_n

        # update posterior of alpha
        """
        α_n = α_0 + (n * d)/2
        """
        self.alpha_n = self.alpha0 + (0.5 * n * self.d)

        # update posterior of beta
        """
        β_n = β_0 + 1/2 SSE + (κ_0 * n)/(2 κ_n) ||x̄ - μ0||^2
        """
        diff = x_bar - self.mu0 # difference between sample mean and prior mean
        self.beta_n = self.beta0 + ( 0.5 * sse ) + ((self.kappa0 * n) / (2.0 * self.kappa_n) * torch.dot(diff, diff))

    def post_params(self):
        """
        Return posterior parameters.
        """
        return {
            "mu_n": self.mu_n,
            "kappa_n": self.kappa_n,
            "alpha_n": self.alpha_n,
            "beta_n": self.beta_n,
        }

    def sample_post_dist(self):
        """
        Sample from the posterior distribution over the mean vector and covariance matrix.
        """

        # Sample inverse-Gamma distribution
        """
        If Y ~ Gamma(alpha, rate=beta),
        then 1/Y ~ InvGamma(alpha, beta)
        """
        prec = D.Gamma(concentration=self.alpha_n, rate=self.beta_n).sample()
        sigma2 = 1.0 / prec
        # Sample mean conditional on sigma^2.
        covariance = (sigma2 / self.kappa_n) * torch.eye(self.d)
        """
        μ ~ 𝒩(μ_n, (σ^2 / κ_n) I)
        """
        mu = D.MultivariateNormal(loc=self.mu_n, covariance_matrix=covariance).sample()

        return mu, sigma2

    def _pred_dist_params(self):
        """
        Parameters of the multivariate Student-t predictive distribution.
        """
        df = 2.0 * self.alpha_n
        scale = ((self.beta_n / self.alpha_n) * (self.kappa_n + 1.0) / self.kappa_n ) * torch.eye(self.d)

        return {
            "df": df,
            "loc": self.mu_n,
            "scale": scale,
        }

    def pred_lh(self, x: torch.Tensor):
        """
        Returns the predictive likelihood of a new observation x.
        """
        params = self._pred_dist_params()
        pred_dist = pyroD.MultivariateStudentT(df=params["df"], loc=params["loc"], scale_tril=torch.linalg.cholesky(params["scale"]),)
        return torch.exp(pred_dist.log_prob(x))


class SuffStatsRewards(SuffStats):
    """
    Maintains sufficient statistics for the reward model within a state and context.
    Sufficient stats for each action are:
        succ[a] : (soft) count of r=1 outcomes
        fail[a] : (soft) count of r=0 outcomes

    This can support for example K independent Bernoulli reward models, or a one-optimal-arm model where the optimal arm is unknown.
    """

    def __init__(self, K: int):
        self.K = K
        self.succ = torch.zeros(K)
        self.fail = torch.zeros(K)

    def update(self, a: int, r: float, confidence: float = 1.0):
        """
        Update with one observation (a, r), where r in {0.0, 1.0}.
        """
        assert 0 <= a < self.K, f"action {a} out of range for {self.K} actions"
        assert (r == 0.0) or (r == 1.0), f"reward r must be 0.0 or 1.0, got {r}"

        if r == 1.0:
            self.succ[a] += confidence
        else:
            self.fail[a] += confidence

    def as_tensors(self):
        return self.succ.clone(), self.fail.clone()

    @property
    def n(self):
        """
        Total (soft) number of observations across all actions.
        """
        return float((self.succ + self.fail).sum().item())

    def to_state(self):
        """
        Minimal state representation needed to reconstruct the sufficient statistics
        """
        return {
            "succ": self.succ, 
            "fail": self.fail
            }

    @classmethod
    def from_state(cls, state):
        """
        Reconstruct sufficient statistics from its minimal state representation
        """
        K = int(state["succ"].numel())
        obj = cls(K)
        obj.succ = state["succ"].clone()
        obj.fail = state["fail"].clone()
        return obj


class ConjugateBernoulli(ConjugateModel):
    """
    Conjugate model for:
        r | a ~ Bernoulli(p_a)
        p_a ~ Beta(alpha0[a], beta0[a])
    independently for each action a in {0, ..., K-1}.
    """

    SUFFSTATS_CLS = SuffStatsRewards

    def __init__(self, alpha0, beta0):
        """
        alpha0:     [K] prior pseudo-counts of successes (r=1) per action
        beta0:      [K] prior pseudo-counts of failures  (r=0) per action
        """
        assert isinstance(alpha0, torch.Tensor) and isinstance(beta0, torch.Tensor), "alpha0 and beta0 must be torch tensors"
        assert (alpha0.dim() == 1 and beta0.dim() == 1), "alpha0 and beta0 must be 1D tensors"
        assert alpha0.shape == beta0.shape, "alpha0 and beta0 must have the same shape"
        assert torch.all(alpha0 > 0), "alpha0 must be > 0"
        assert torch.all(beta0 > 0), "beta0 must be > 0"

        self.alpha0 = alpha0.clone().to(torch.float)
        self.beta0  = beta0.clone().to(torch.float)
        self.K      = int(alpha0.shape[0])

        self.suffstats = SuffStatsRewards(self.K)
        self.reset_posterior()

    def reset_posterior(self):
        """
        Posterior parameters start at prior.
        """
        self.alpha_n = self.alpha0.clone()
        self.beta_n  = self.beta0.clone()

    def update(self, a: int, r: float, confidence: float = 1.0):
        """
        Update sufficient stats with (a, r) and refresh posterior parameters.
        """
        self.suffstats.update(a, r, confidence=confidence)
        self._update_posterior()

    def _update_posterior(self):
        """
        Beta posterior for each action:
            alpha_n[a] = alpha0[a] + succ[a]
            beta_n[a]  = beta0[a]  + fail[a]
        """

        if self.suffstats.n == 0: # do not update posterior if there are no datapoints
            return

        succ, fail = self.suffstats.as_tensors()
        self.alpha_n = self.alpha0 + succ.to(torch.float)
        self.beta_n = self.beta0 + fail.to(torch.float)

    def post_params(self):
        return {
            "alpha_n": self.alpha_n, 
            "beta_n":  self.beta_n
            }

    def sample_post_dist(self):
        """
        Returns a sample of the reward probabilities for each action from the posterior Beta distribution.
        """
        return D.Beta(self.alpha_n, self.beta_n).sample()

    def _pred_dist_params(self):
        """
        Returns parameters of the Bernoulli predictive distribution for each action.
        """
        p_rew = self.alpha_n / (self.alpha_n + self.beta_n)

        return {
            "p_rew": p_rew
        }
    
    def pred_lh(self, a: int, r: float):
        """
        Predictive likelihood of observing reward r given action a.
            P(r=1 | a) = alpha_n[a] / (alpha_n[a] + beta_n[a])
            P(r=0 | a) = 1 - P(r=1 | a)
        """
        p = self._pred_dist_params()["p_rew"][a]
        if r == 1.0:
            return p
        elif r == 0.0:
            return 1.0 - p
        else:
            raise ValueError(f"reward r must be 0.0 or 1.0, got {r}")

    



class ConjugateOptimalArm(ConjugateModel):
    """
    Bayesian one-optimal-arm model, in which there is exactly one optimal arm k.
    Conditional on k:

        P(r=1 | a=k)   = rho_c
        P(r=1 | a!=k)  = rho_i

    The prior probability that arm k is optimal is beta0[k].

    Given sufficient statistics (succ, fail),

        P(k | data) proportional to

            beta0[k]
            * (rho_c / rho_i) ** succ[k]
            * ((1-rho_c) / (1-rho_i)) ** fail[k].
    """

    SUFFSTATS_CLS = SuffStatsRewards

    def __init__(self, rho_c, rho_i, beta0):
        """
        rho_c:      probability of reward for the optimal arm
        rho_i:      probability of reward for the non-optimal arms
        beta0:      prior categorical distribution over the optimal arm, shape [K], sums to 1.0
        """

        assert 0.0 < rho_c < 1.0, f"rho_c must be in (0, 1), got {rho_c}"
        assert 0.0 < rho_i < 1.0, f"rho_i must be in (0, 1), got {rho_i}"
        self.rho_c = torch.tensor(rho_c)
        self.rho_i = torch.tensor(rho_i)

        assert isinstance(beta0, torch.Tensor), "beta0 must be a torch tensor"
        assert beta0.ndim == 1, "beta0 must be a 1D tensor"
        assert beta0.numel() >= 2, "beta0 must contain at least two arms"
        assert torch.all(beta0 > 0), "all beta0 entries must be > 0 to avoid degenerate posteriors"
        assert torch.sum(beta0) == 1.0, "beta0 must sum to 1.0"
        self.beta0 = beta0.clone()
        self.K = beta0.numel()

        self.suffstats = SuffStatsRewards(self.K)

        # These are constant across all posterior updates.
        self.log_rho_ratio     = torch.log(self.rho_c / self.rho_i)
        self.log_failure_ratio = torch.log((1.0 - self.rho_c) / (1.0 - self.rho_i))

        self.log_beta0 = self.beta0.log()

        self.reset_posterior()

    def reset_posterior(self):
        """
        Reset the posterior to the prior.
        """
        self.beta_n = self.beta0.clone()

    def update(self, a: int, r: float, confidence: float = 1.0):
        """
        Incorporate observation (a, r) and recompute the posterior.
        """
        self.suffstats.update(a, r, confidence=confidence)
        self._update_posterior()

    def _update_posterior(self):
        """
        Compute the posterior over the identity of the optimal arm:
            log P(k | data) + log beta0[k] + succ[k] * log(rho_c / rho_i) + fail[k] * log((1-rho_c)/(1-rho_i)) + constant 
        """
        succ, fail = self.suffstats.as_tensors()
        self.beta_n = torch.softmax(self.log_beta0 + (succ * self.log_rho_ratio) + (fail * self.log_failure_ratio), dim=0)

    def post_params(self):
        """
        Posterior categorical distribution over the optimal arm.
        """
        return {
            "beta_n": self.beta_n
        }

    def sample_post_dist(self):
        """
        Posterior sample of the optimal arm identity, drawn from the categorical distribution defined by beta_n.
        """
        best_arm = D.Categorical(probs=self.beta_n).sample()
        arms = torch.zeros(self.K)
        arms[best_arm] = 1.0

        return arms

    def _pred_dist_params(self):
        """Parameters of the Bernoulli predictive distributions."""

        p_rew = self.beta_n * self.rho_c + (1.0 - self.beta_n) * self.rho_i

        return {
            "p_rew": p_rew
        }

    def pred_lh(self, a: int, r: float):
        """
        Predictive likelihood of observing reward r given action a.
            P(r=1 | a) = alpha_n[a] / (alpha_n[a] + beta_n[a])
            P(r=0 | a) = 1 - P(r=1 | a)
        """
        p = self._pred_dist_params()["p_rew"][a]

        if r == 1.0:
            return p
        elif r == 0.0:
            return 1.0 - p
        else:
            raise ValueError(f"reward r must be 0.0 or 1.0, got {r}")
