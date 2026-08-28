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
        raise NotImplementedError(
            "sample_post_dist method must be implemented by subclass"
        )

    def _pred_dist_params(self):
        """
        Return parameters of the posterior predictive distribution.
        """
        raise NotImplementedError(
            "_pred_dist_params method must be implemented by subclass"
        )

    def pred_lh(self, x):
        """
        Return the predictive likelihood of a new observation x.
        """
        raise NotImplementedError("pred_lh method must be implemented by subclass")

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
        self.n = 0
        self.sum_x = torch.zeros(d)
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
            "n": float(self.n),
            "sum_x": self.sum_x,
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


class ConjugateGaussian(ConjugateModel):
    SUFFSTATS_CLS = SuffStatsGaussian

    def __init__(self, mu0, kappa0, nu0, Lambda0):
        """
        Conjugate Gaussian-Inverse Wishart prior for multivariate Gaussian with unknown mean & covariance.

        mu0:        prior mean mean [d]
        kappa0:     pseudocount of prior measurements
        nu0:        degrees of freedom > d-1
        Lambda0:    scale matrix [d,d], positive definite
        """
        assert type(mu0) == torch.Tensor, "prior mean mu0 must be a torch tensor"
        assert mu0.dim() == 1, "prior mean mu0 must be a 1D tensor"
        self.mu0 = mu0

        assert (
            type(kappa0) == float
        ), "pseudocount of prior measurements kappa0 must be a float"
        assert kappa0 > 0, "pseudocount of prior measurements kappa0 must be > 0"
        self.kappa0 = float(kappa0)

        assert type(nu0) == float, "nu0 must be a float"
        assert (
            nu0 > mu0.shape[0] - 1
        ), f"degrees of freedom nu0 must be > {mu0.shape[0]-1}"
        self.nu0 = float(nu0)

        assert (
            type(Lambda0) == torch.Tensor
        ), "scale matrix Lambda0 must be a torch tensor"
        assert Lambda0.dim() == 2, "scale matrix Lambda0 must be a 2D tensor"
        assert (
            Lambda0.shape[0] == Lambda0.shape[1] == mu0.shape[0]
        ), "scale matrix Lambda0 must be of shape (d,d)"
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
        if self.suffstats.n == 0:  # do not update posterior if there are no datapoints
            return

        x_bar = self.suffstats.mean
        n = self.suffstats.n

        # Update posterior of mean
        """
        μ_n = ( (κ_0 / (κ_0 + n)) * μ0 ) + ( (n / (κ0 + n)) * x̄ )
        """
        self.mu_n = ((self.kappa0 / (self.kappa0 + n)) * self.mu0) + (
            (n / (self.kappa0 + n)) * x_bar
        )

        # Update posterior of kappa
        """
        κ_n = κ_0 + n
        """
        self.kappa_n = self.kappa0 + n

        # Update posterior of nu
        """
        ν_n = ν_0 + n
        """
        self.nu_n = self.nu0 + n

        # Update posterior of Lambda
        """
        Λ_n = Λ_0 + S + ((κ_0 * n )/ (κ_n)) * (x̄ - μ0)(x̄ - μ0)^T
        """
        S = self.suffstats.scatter
        diff = (x_bar - self.mu0).unsqueeze(
            1
        )  # column vector, difference between sample mean and prior mean
        self.Lambda_n = (
            self.Lambda0 + S + ((self.kappa0 * n) / (self.kappa_n)) * (diff @ diff.T)
        )

    def post_params(self):
        return {
            "mu_n": self.mu_n,
            "kappa_n": self.kappa_n,
            "nu_n": self.nu_n,
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
        Prec = D.Wishart(
            df=self.nu_n, covariance_matrix=torch.linalg.inv(self.Lambda_n)
        ).sample()
        Sigma = torch.linalg.inv(Prec)
        # Sample mean from Gaussian
        """
        μ ~ 𝒩(μ_n, Σ / κ_n)
        """
        mu = D.MultivariateNormal(
            loc=self.mu_n, covariance_matrix=(Sigma / self.kappa_n)
        ).sample()

        return mu, Sigma

    def _pred_dist_params(self):
        """
        Returns parameters of the multivariate Student-t predictive distribution (uncerteanty over mean and variance is integrated out).
        """
        df = self.nu_n - self.d + 1
        scale = (self.Lambda_n * (self.kappa_n + 1)) / (self.kappa_n * df)

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
        pred_dist = pyroD.MultivariateStudentT(
            df=params["df"],
            loc=params["loc"],
            scale_tril=torch.linalg.cholesky(params["scale"]),
        )
        return torch.exp(pred_dist.log_prob(x))


class SuffStatsBernoulli(SuffStats):
    """
    Maintains sufficient statistics for a collection of K Bernoulli distributions.

    Sufficient stats for each action a:
        succ[a] : (soft) count of r=1 outcomes
        fail[a] : (soft) count of r=0 outcomes

    This supports confidence-weighted (fractional) updates.
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
        return {"succ": self.succ, "fail": self.fail}

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

    SUFFSTATS_CLS = SuffStatsBernoulli

    def __init__(self, alpha0: torch.Tensor, beta0: torch.Tensor):
        """
        alpha0:     [K] prior pseudo-counts of successes (r=1) per action
        beta0:      [K] prior pseudo-counts of failures  (r=0) per action
        """
        assert isinstance(alpha0, torch.Tensor) and isinstance(
            beta0, torch.Tensor
        ), "alpha0 and beta0 must be torch tensors"
        assert (
            alpha0.dim() == 1 and beta0.dim() == 1
        ), "alpha0 and beta0 must be 1D tensors"
        assert alpha0.shape == beta0.shape, "alpha0 and beta0 must have the same shape"
        assert torch.all(alpha0 > 0), "alpha0 must be > 0"
        assert torch.all(beta0 > 0), "beta0 must be > 0"

        self.alpha0 = alpha0.clone().to(torch.float)
        self.beta0 = beta0.clone().to(torch.float)
        self.K = int(alpha0.shape[0])

        self.suffstats = SuffStatsBernoulli(self.K)
        self.reset_posterior()

    def reset_posterior(self):
        """
        Posterior parameters start at prior.
        """
        self.alpha_n = self.alpha0.clone()
        self.beta_n = self.beta0.clone()

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

        if self.suffstats.n == 0:  # do not update posterior if there are no datapoints
            return

        succ, fail = self.suffstats.as_tensors()
        self.alpha_n = self.alpha0 + succ.to(torch.float)
        self.beta_n = self.beta0 + fail.to(torch.float)

    def post_params(self):
        return {"alpha_n": self.alpha_n, "beta_n": self.beta_n}

    def predictive_p(self):
        """
        Posterior predictive mean of p_a = P(r=1 | a) for each action.
        """
        return self.alpha_n / (self.alpha_n + self.beta_n)

    def _pred_dist_params(self):
        """
        Returns parameters of the Bernoulli predictive distribution for each action.
        """
        return self.predictive_p()

    def pred_lh(self, a: int, r: float):
        """
        Predictive likelihood of observing reward r given action a.
            P(r=1 | a) = alpha_n[a] / (alpha_n[a] + beta_n[a])
            P(r=0 | a) = 1 - P(r=1 | a)
        """
        p = self.predictive_p()[a]
        if r == 1.0:
            return p
        elif r == 0.0:
            return 1.0 - p
        else:
            raise ValueError(f"reward r must be 0.0 or 1.0, got {r}")

    def sample_post_dist(self):
        """
        Returns a tensor of shape [K] with one sampled p_a per action.
        """
        return D.Beta(self.alpha_n, self.beta_n).sample()


class SuffStatsCategorical(SuffStats):
    """
    Maintains sufficient statistics for a categorical distribution.
    Sufficient stats: counts of each category.
    """

    def __init__(self, K: int):
        self.K = K
        self.counts = torch.zeros(K)

    def update(self, x: int, confidence: float = 1.0):
        """
        Update with a new categorical observation (int in [0, K-1]).
        """
        assert 0 <= x < self.K, f"observation {x} out of range for {self.K} categories"
        self.counts[x] += confidence

    @property
    def n(self):
        return int(self.counts.sum())

    def as_tensor(self):
        return self.counts.clone()

    def to_state(self):
        """
        Minimal state representation needed to reconstruct the sufficient statistics.
        """
        return {"counts": self.counts}

    @classmethod
    def from_state(cls, state):
        """
        Reconstruct sufficient statistics from its minimal state representation.
        """
        K = int(state["counts"].numel())
        obj = cls(K)
        obj.counts = state["counts"].clone()
        return obj


class ConjugateCategorical(ConjugateModel):
    """
    Conjugate model: categorical likelihood with Dirichlet prior.
    Supports incremental updates and posterior predictive probabilities.
    """

    SUFFSTATS_CLS = SuffStatsCategorical

    def __init__(self, alpha0: torch.Tensor):
        assert alpha0.dim() == 1, "alpha0 must be a 1D tensor"
        assert torch.all(alpha0 > 0), "alpha0 parameters must be > 0"

        self.alpha0 = alpha0.clone().to(torch.float)
        self.K = alpha0.shape[0]

        self.suffstats = SuffStatsCategorical(self.K)

        # Start at prior
        self.reset_posterior()

    def reset_posterior(self):
        self.alpha_n = self.alpha0.clone()

    def update(self, x: int, confidence: float = 1.0):
        """
        Add a new observation and update posterior parameters.
        """
        self.suffstats.update(x, confidence)
        self._update_posterior()

    def _update_posterior(self):
        self.alpha_n = self.alpha0 + self.suffstats.as_tensor().to(torch.float)

    def post_params(self):
        return {"alpha_n": self.alpha_n}

    def sample_post_dist(self):
        """
        Sample from the posterior distribution over theta.
        """
        theta = D.Dirichlet(self.alpha_n).sample()
        return theta

    def _pred_dist_params(self):
        """
        Posterior predictive probabilities for each category:
        theta_n = (alpha_n / sum(alpha_n))
        """
        theta_n = self.alpha_n / self.alpha_n.sum()
        return theta_n

    def pred_lh(self, x: int):
        """
        Predictive likelihood of a new categorical observation.
        """
        probs = self._pred_dist_params()
        return probs[x].item()


class SuffStatsWinningArm(SuffStats):
    """
    Sufficient statistics for a bandit with exactly one correct/winning arm.

    For each action a:
        succ[a] : soft count of rewards r=1
        fail[a] : soft count of rewards r=0

    Unlike SuffStatsBernoulli, these statistics are not interpreted as
    independent Bernoulli distributions. They are observations used to
    infer the single latent winning arm.
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
        return float((self.succ + self.fail).sum().item())

    def to_state(self):
        return {
            "succ": self.succ,
            "fail": self.fail,
        }

    @classmethod
    def from_state(cls, state):
        K = int(state["succ"].numel())
        obj = cls(K)
        obj.succ = state["succ"].clone()
        obj.fail = state["fail"].clone()
        return obj


class ConjugateWinningArm(ConjugateModel):
    """
    Bayesian model for a K-armed bandit with exactly one winning/correct arm.

    Latent variable:
        C ~ Categorical(prior)

    Observation model:

        r | a, C=a     ~ Bernoulli(p_c)
        r | a, C!=a    ~ Bernoulli(p_i)

    Thus the arms are NOT independent Bernoulli distributions.

    Instead, there is a single categorical posterior over which arm is
    the correct arm.
    """

    SUFFSTATS_CLS = SuffStatsWinningArm

    def __init__(
        self,
        p_c: float,
        p_i: float,
        prior: torch.Tensor,
    ):
        """
        Args:
            p_c:
                P(r=1 | selected arm is the correct arm)

            p_i:
                P(r=1 | selected arm is NOT the correct arm)

            prior:
                [K] prior probabilities over which arm is correct.
                For a uniform prior, use torch.ones(K) / K.
        """
        assert isinstance(prior, torch.Tensor), "prior must be a torch tensor"
        assert prior.dim() == 1, "prior must be a 1D tensor"
        assert torch.all(prior >= 0), "prior must be >= 0"
        assert torch.isclose(
            prior.sum(),
            torch.tensor(1.0, dtype=prior.dtype),
            atol=1e-5,
        ), "prior must sum to 1"

        assert 0.0 <= p_c <= 1.0, "p_c must be in [0, 1]"
        assert 0.0 <= p_i <= 1.0, "p_i must be in [0, 1]"

        self.p_c = float(p_c)
        self.p_i = float(p_i)

        self.prior = prior.clone().to(torch.float)
        self.K = int(prior.shape[0])

        self.suffstats = SuffStatsWinningArm(self.K)
        self.reset_posterior()

    def reset_posterior(self):
        """
        Reset posterior to the prior.
        """
        self.posterior = self.prior.clone()

    def update(self, a: int, r: float, confidence: float = 1.0):
        """
        Update posterior with observation (a, r).

        The update is Bayesian:

            posterior(C=k)
                ∝ prior(C=k) P(r | a, C=k)

        For k=a:
            P(r=1) = p_c
            P(r=0) = 1-p_c

        For k!=a:
            P(r=1) = p_i
            P(r=0) = 1-p_i

        `confidence` implements a fractional likelihood update.
        """
        self.suffstats.update(a, r, confidence=confidence)
        self._update_posterior()

    def _update_posterior(self):
        """
        Recompute:

            P(C=k | data)
                ∝ P(C=k)
                   * product_t P(r_t | a_t, C=k)

        using the sufficient statistics.
        """
        succ, fail = self.suffstats.as_tensors()

        # Log posterior for each possible winning arm.
        log_post = torch.log(self.prior)

        for k in range(self.K):
            # Observations obtained when pulling k:
            # these use p_c under hypothesis C=k.
            log_post[k] += succ[k] * torch.log(torch.tensor(self.p_c)) + fail[
                k
            ] * torch.log(torch.tensor(1.0 - self.p_c))

            # Observations obtained from all other arms:
            # these use p_i under hypothesis C=k.
            other_succ = succ.sum() - succ[k]
            other_fail = fail.sum() - fail[k]

            log_post[k] += other_succ * torch.log(
                torch.tensor(self.p_i)
            ) + other_fail * torch.log(torch.tensor(1.0 - self.p_i))

        # Normalize in log space for numerical stability.
        self.posterior = torch.softmax(log_post, dim=0)

    def post_params(self):
        """
        Return posterior probabilities for each arm being the winner.
        """
        return {
            "posterior": self.posterior.clone(),
        }

    def predictive_p(self):
        """
        Posterior predictive probability of reward for each action.

        If action a is selected:

            P(r=1 | a, data)
                = P(C=a | data) * p_c
                  + P(C!=a | data) * p_i

        Therefore:

            p(a) = p_i + (p_c - p_i) * P(C=a | data)
        """
        return self.p_i + (self.p_c - self.p_i) * self.posterior

    def _pred_dist_params(self):
        """
        Returns Bernoulli predictive probability for each action.
        """
        return self.predictive_p()

    def pred_lh(self, a: int, r: float):
        """
        Predictive likelihood of observing reward r given action a.
        """
        assert 0 <= a < self.K, f"action {a} out of range for {self.K} actions"

        p = self.predictive_p()[a]

        if r == 1.0:
            return p
        elif r == 0.0:
            return 1.0 - p
        else:
            raise ValueError(f"reward r must be 0.0 or 1.0, got {r}")

    def sample_post_dist(self):
        """
        Sample the winning arm from the posterior.

        Returns:
            Scalar tensor containing the sampled winning arm index.
        """
        return D.Categorical(probs=self.posterior).sample()
