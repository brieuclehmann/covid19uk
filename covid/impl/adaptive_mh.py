""" Multivariate Normal Random Walk Transition Kernel with adaptive step size. """
""" Authors: Alison Hale and Chris Jewell """
""" Version: 0.0.5 """

import collections

from tensorflow_probability.python import stats
from tensorflow_probability.python.distributions.bernoulli import Bernoulli
from tensorflow_probability.python.distributions.mvn_tril import MultivariateNormalTriL
from tensorflow_probability.python.distributions.normal import Normal
from tensorflow_probability.python.experimental import (
    stats,
)  # ***tfp nightly experimental***
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.util.seed_stream import SeedStream


__all__ = [
    "UncalibratedMVNRandomWalkResults",
    "CovarianceStepAdaptationExtraResults",
    "rwm_extra_setter_fn",
    "rwm_results_getter_fn" "random_walk_mvnorm_fn",
    "CovarianceStepAdaptation",
    "UncalibratedMVNRandomWalk",
]


class UncalibratedMVNRandomWalkResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        "UncalibratedMVNRandomWalkResults",
        ["log_acceptance_correction", "target_log_prob", "is_adaptive",],
    ),
):
    """Internal state for Uncalibrated MVN random walk.
    
    Attributes:
      log_acceptance_correction: `Tensor` representing the `log`
        acceptance-correction.
      target_log_prob: Floating point scalar `Tensor`. Target accept
        probability.
      is_adaptive: Integer scalar representing if the adaptive or
        fixed proposal was used to purturb the current state.
    """

    __slots__ = ()


class CovarianceStepAdaptationExtraResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        "CovarianceStepAdaptationExtraResults",
        ["num_steps", "covariance_scaling", "covariance", "running_covariance",],
    ),
):
    """State information for `MetropolisHastings` `extra` attribute.

    Attributes:
      num_steps: Python integer representing the current number of
        `MetropolisHastings` steps.
      covariance_scaling: Python floating point number representing a 
        the value of the prefactor which is used at each 
        `covariance_update_inverval`. The `covariance_scaling`
        is tuned during the evolution of the MCMC chain.  Let d represent the
        number of parameters e.g. as given by the initial_state. The ratio given
        by the `covariance_scaling` divided by d is used to multiply the running 
        covariance at each `covariance_update_inverval`.  The result is an
        updated covariance matrix which is used as the proposal during the
        current `covariance_update_inverval`.
        Default value: 2.38**2.
      covariance: Python `list` of `Tensor`s representing the current 
        covariance of the proposal.
      running_covariance: running covariance state as stored by 
        the instantiation of `RunningCovariance` from `stats`.
    """

    __slots__ = ()


@mcmc_util.make_innermost_setter
def rwm_extra_setter_fn(
    kernel_results, num_steps, covariance_scaling, covariance, running_covariance
):
    """Setter for `extra` member of `MetropolisHastings` `TransitionKernel` so
  that it can be adapted."""
    # Note that the `make_innermost_setter` wrapper above handles the relatively
    # common case of that member being called `inner_results`.
    return kernel_results._replace(
        extra=CovarianceStepAdaptationExtraResults(
            num_steps=num_steps,
            covariance_scaling=covariance_scaling,
            covariance=covariance,
            running_covariance=running_covariance,
        )
    )


@mcmc_util.make_innermost_getter
def rwm_results_getter_fn(kernel_results):
    """Getter for `extra` member of `MetropolisHastings` `TransitionKernel` so
  that it can be inspected."""
    # Note that the `make_innermost_getter` wrapper above handles the relatively
    # common case of that member being called `inner_results`.
    return kernel_results


def random_walk_mvnorm_fn(covariance, pu=0.95, fixed_variance=0.01, name=None):
    """Returns callable that adds Multivariate Normal (MVN) noise to the input.
  
  Args:
    covariance: Python `list` of `Tensor`s representing each covariance 
      matrix, size d x d, of the Multivariate Normal proposal.
    pu: Python floating point number representing the bounded convergence
      parameter. If equal to 1, then all proposals are drawn
      from the MVN(0, `covariance`) distribution, if less than 1, 
      proposals are drawn from MVN(0, covariance) with probability pu, and 
      MVN(0, `fixed_variance`/d) otherwise.
      Default value: 0.95.
    fixed_variance: Python floating point number representing the variance of
      the fixed proposal distribution of the form MVN(0, `fixed_variance`/d).
      Default value: 0.01.
    name: Python `str` name.
      Given the default value of `None` the name is set to `random_walk_mvnorm_fn`.
 
  Returns:
      random_walk_mvnorm_fn: A callable accepting a Python `list` of `Tensor`s
      representing the state parts of the `current_state` and an `int`
      representing the random seed to be used to generate the proposal. The
      callable returns two quantities. First, a `Tensor` of type integer
      representing whether each state part was updated using the fixed
      (value=0) or adaptive (value=1) proposal.  Second, a `list` of 
      `Tensor`s, with the same-type as the input state parts, which represents
      the proposal for the Metropolis Hastings algorithm.
  """

    dtype = dtype_util.base_dtype(covariance[0].dtype)
    shape = tf.stack(covariance, axis=0).shape
    # for numerical stability ensure covariance matrix is positive semi-definite
    covariance = covariance + 1.0e-9 * tf.eye(
        shape[1], batch_shape=[shape[0]], dtype=dtype
    )
    scale_tril = tf.linalg.cholesky(covariance)
    rv_adaptive = MultivariateNormalTriL(
        loc=tf.zeros([shape[0], shape[1]], dtype=dtype), scale_tril=scale_tril
    )
    rv_fixed = Normal(
        loc=tf.zeros([shape[0], shape[1]], dtype=dtype),
        scale=tf.constant(fixed_variance, dtype=dtype) / shape[2],
    )
    probs = tf.expand_dims(tf.ones([shape[0]], dtype=dtype) * pu, axis=1)
    u = Bernoulli(probs=probs)

    def _fn(state_parts, seed):
        with tf.name_scope(name or "random_walk_mvnorm_fn"):
            seed_stream = SeedStream(seed, salt="RandomWalkMvnormFn")

            def proposal():
                # For parallel computation it is quicker to sample
                # both distributions then select the result
                rv = tf.stack(
                    [
                        rv_fixed.sample(seed=seed_stream),
                        rv_adaptive.sample(seed=seed_stream),
                    ],
                    axis=1,
                )
                uv = u.sample(seed=seed_stream)
                return uv, tf.squeeze(tf.gather(rv, uv, axis=1, batch_dims=1), axis=1)

            uv, proposal = proposal()
            proposal_parts = tf.unstack(proposal)
            new_state_parts = [
                proposal_part + state_part
                for proposal_part, state_part in zip(proposal_parts, state_parts)
            ]
            return uv, new_state_parts

    return _fn


class CovarianceStepAdaptation(kernel_base.TransitionKernel):
    """Adaptive Multisite Random Walk Metropolis Hastings Algorithm.
    Consider a continuous multivariate random variable X of dimension d,
    distributed according to a probability distribution function pi(x) 
    known up to a normalising constant. The general principles are
    outlined by Roberts and Rosenthal (2009)][1].  Specifically we follow
    Algorithm 6 of Sherlock et al. (2010)[2], in which we update the MCMC
    chain by proposing from a multivariate Normal random variable, adapting
    both the variance and correlation structure of the covariance matrix.

    In pseudo code the algorithm is:
    ```
    Inputs:
      i, j iteration indices with initial values 0
      d number of dimensions (i.e. number of parameters)
      N total number of steps
      X[0] initial chain state
      S = 0.001.eye(d) = initial covariance matrix
      m[0] = 2.38^2/d the initial variance scalar i.e. covariance_scaling/d
      t = 0.234 = target_accept_ratio
      l = 0.01 = covariance_scaling_limiter
      k = 0.7 = covariance_scaling_reducer
      pu = 0.95
      f = 0.01 = fixed_variance
      covariance_burnin = 100
      covariance_update_inverval = 100
      pi(.) denotes probability distribution of argument

    for i = 0,...,N do  

      // Adapt covariance_scaling, m
      if u[i-1] < pu then // Only adapt if adaptive part was proposed (NB u[-1]=1.0)
        Let alpha[i-1] = min(1, pi(X*)/pi(X[i-1])
        Update m[i] = m[i-1] . exp[ sgn(alpha[i-1] − t) . min(l, (i-1)^(−k)) ]
      end
      
      // Adapt covariance matrix, S
      if i>covariance_burnin and mod(i,covariance_update_inverval)==0 then
        Update j = j + 1
        Update S[j] = Cov(X[0,...,i])
      end

      // Propose new state
      Draw u[i] ~ Uniform(0, 1)
      if u[i] < pu then
        // Adaptive part
        Draw X* ~ MVN(X[i], m[i] . S[j])
      else                            
        // Fixed part
        Draw X* ~ MVN(X[i], f . IdentityMatrix / d)
      end

      // Perform MH accept/reject
      Let alpha[i] = min(1, pi(X*)/pi(X[i])
      Draw v ~ Uniform(0, 1)
      if v < alpha[i] then
        Update X[i+1] = X*
      else
        Update X[i+1] = X[i]
      end

      Update i = i + 1

    end
    ```

    #### Example
    ```python

    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp  # 10-09-2020 use nightly

    tfd = tfp.distributions

    dtype = np.float32

    # data
    x = dtype([-2.11, -0.81, 3.29, -3.11, -2.41, -4.01, 3.39, 3.59, 2.89, -0.71])
    y = dtype([6.21, 7.79, 8.09, 2.70, 4.79, 2.41, 10.69, 9.01, 9.59, 5.71])

    # define linear regression model
    def Model(x):
      def alpha():
        return tfd.Normal(loc=dtype(0.), scale=dtype(1000.))
      def beta():
        return tfd.Normal(loc=dtype(0.), scale=dtype(100.))
      def sigma():
        return tfd.Gamma(concentration=dtype(0.1), rate=dtype(0.1))
      def y(alpha, beta, sigma):
        mu = alpha + beta * x
        return tfd.Normal(mu, scale=sigma)
      return tfd.JointDistributionNamed(dict(
        alpha = alpha,
        beta = beta,
        sigma = sigma,
        y = y))

    # target log probability of linear model
    def log_prob(param):
      alpha, beta, sigma = tf.unstack(param, axis=-1)
      lp = model.log_prob({'alpha': alpha, 
                           'beta': beta, 
                           'sigma': sigma, 
                           'y': y})
      return tf.reduce_sum(lp)

    # posterior distribution MCMC chain
    @tf.function
    def posterior(iterations, burnin, thinning, initial_state):
      kernel = CovarianceStepAdaptation(
        target_log_prob_fn = log_prob,
        initial_state = initial_state)

      return tfp.mcmc.sample_chain(
        num_results = iterations,
        current_state = initial_state,
        kernel = kernel,
        num_burnin_steps = burnin,
        num_steps_between_results = thinning,
        parallel_iterations = 1,
        trace_fn = lambda state, results: results)

    # initialize model
    model = Model(x)
    initial_state = dtype([0.1, 0.1, 0.1]) # start chain at alpha=0.1, beta=0.1, sigma=0.1

    # estimate posterior distribution
    samples, results = posterior(
      iterations = 1000,
      burnin = 0,
      thinning = 0,
      initial_state = initial_state)

    tf.print('\nAcceptance probability:',
      tf.math.reduce_mean(
        tf.cast(results.is_accepted, dtype=tf.float32)))
    tf.print('\nalpha samples:', samples[0])
    tf.print('\nbeta  samples:', samples[1])
    tf.print('\nsigma samples:', samples[2])

    ```

    #### References

    [1]: Gareth Roberts, Jeffrey Rosenthal. Examples of Adaptive MCMC.
         _Journal of Computational and Graphical Statistics_, 2009.
       http://probability.ca/jeff/ftpdir/adaptex.pdf

    [2]: Chris Sherlock, Paul Fearnhead, Gareth O. Roberts. The Random
         Walk Metropolis: Linking Theory and Practice Through a Case Study. 
       _Statistical Science_, 25:172–190, 2010.
       https://projecteuclid.org/download/pdfview_1/euclid.ss/1290175840

    """

    def __init__(
        self,
        target_log_prob_fn,
        initial_state,
        initial_covariance=None,
        initial_covariance_scaling=2.38 ** 2,
        covariance_scaling_reducer=0.7,
        covariance_scaling_limiter=0.01,
        covariance_burnin=100,
        covariance_update_interval=100,
        target_accept_ratio=0.234,
        pu=0.95,
        fixed_variance=0.01,
        seed=None,
        name=None,
    ):
        """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` and returns its (possibly unnormalized) log-density
        under the target distribution.
      initial_state: Python `list` of `Tensor`s representing the initial
        state of each parameter.
      initial_covariance: Python `list` of `Tensor`s representing the
        initial covariance of the proposal. The `initial_covariance` and 
        `initial_state` should have identical `dtype`s and 
        batch dimensions.  If `initial_covariance` is `None` then it 
        initialized to a Python `list` of `Tensor`s where each tensor is 
        the identity matrix multiplied by 0.001; the `list` structure will
        be identical to `initial_state`. The covariance matrix is tuned
        during the evolution of the MCMC chain.
        Default value: `None`.
      initial_covariance_scaling: Python floating point number representing a 
        the initial value of the `covariance_scaling` which is used at each 
        `covariance_update_interval`. The value of `covariance_scaling`
        is tuned during the evolution of the MCMC chain.  Let d represent the
        number of parameters e.g. as given by the `initial_state`. The ratio
        given by the `covariance_scaling` divided by d is used to multiply 
        the running covariance at each `covariance_update_interval`.  The 
        result is an updated covariance matrix which is used as the proposal
        during the current `covariance_update_interval`.
        Default value: 2.38**2.
      covariance_scaling_reducer: Python floating point number, bounded over the 
        range (0.5,1.0], representing the constant factor used during the
        adaptation of the `covariance_scaling`. 
        Default value: 0.7.
      covariance_scaling_limiter: Python floating point number, bounded between 
        0.0 and 1.0, which places a limit on the maximum amount the
        `covariance_scaling` value can be purturbed at each interaction of the 
        MCMC chain.
        Default value: 0.01.
      covariance_burnin: Python integer number of steps to take before starting to 
        compute the running covariance.
        Default value: 100.
      covariance_update_interval: Python integer number of steps to take before the
        covariance is next updated when the number of steps is greater than
        or equal to the `covariance_burnin`. Note that the running covariance is
        reset using the current state at this interval. 
        Default value: 100.
      target_accept_ratio: Python floating point number, bounded between 0.0 and 1.0,
        representing the target acceptance probability of the 
        Metropolis–Hastings algorithm.
        Default value: 0.234.
      pu: Python floating point number, bounded between 0.0 and 1.0, representing the 
        bounded convergence parameter.  See `random_walk_mvnorm_fn()` for further
        details.
        Default value: 0.95.
      fixed_variance: Python floating point number representing the variance of
        the fixed proposal distribution. See `random_walk_mvnorm_fn` for 
        further details.
        Default value: 0.01.
      seed: Python integer to seed the random number generator.
        Default value: `None`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None`.

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if `initial_covariance_scaling` is less than or equal
        to 0.0.
      ValueError: if `covariance_scaling_reducer` is less than or equal
        to 0.5 or greater than 1.0.
      ValueError: if `covariance_scaling_limiter` is less than 0.0 or
        greater than 1.0.
      ValueError: if `covariance_burnin` is less than 0.
      ValueError: if `covariance_update_interval` is less than 1.
      ValueError: if `target_accept_ratio` is less than 0.0 or
        greater than 1.0.
      ValueError: if `pu` is less than 0.0 or greater than 1.0.
      ValueError: if `fixed_variance` is less than 0.0.
    """
        with tf.name_scope(
            mcmc_util.make_name(name, "CovarianceStepAdaptation", "__init__")
        ) as name:
            if initial_covariance_scaling <= 0.0:
                raise ValueError(
                    "`{}` must be a `float` greater than 0.0".format(
                        "initial_covariance_scaling"
                    )
                )
            if covariance_scaling_reducer <= 0.5 or covariance_scaling_reducer > 1.0:
                raise ValueError(
                    "`{}` must be a `float` greater than 0.5 and less than or equal to 1.0.".format(
                        "covariance_scaling_reducer"
                    )
                )
            if covariance_scaling_limiter < 0.0 or covariance_scaling_limiter > 1.0:
                raise ValueError(
                    "`{}` must be a `float` between 0.0 and 1.0.".format(
                        "covariance_scaling_limiter"
                    )
                )
            if covariance_burnin < 0:
                raise ValueError(
                    "`{}` must be a `integer` greater or equal to 0.".format(
                        "covariance_burnin"
                    )
                )
            if covariance_update_interval < 1:
                raise ValueError(
                    "`{}` must be a `integer` greater or equal to 1.".format(
                        "covariance_update_interval"
                    )
                )
            if target_accept_ratio < 0.0 or target_accept_ratio > 1.0:
                raise ValueError(
                    "`{}` must be a `float` between 0.0 and 1.0.".format(
                        "target_accept_ratio"
                    )
                )
            if pu < 0.0 or pu > 1.0:
                raise ValueError(
                    "`{}` must be a `float` between 0.0 and 1.0.".format("pu")
                )
            if fixed_variance < 0.0:
                raise ValueError(
                    "`{}` must be a `float` greater than 0.0.".format("fixed_variance")
                )

        if mcmc_util.is_list_like(initial_state):
            initial_state_parts = list(initial_state)
        else:
            initial_state_parts = [initial_state]
        initial_state_parts = [
            tf.convert_to_tensor(s, name="initial_state") for s in initial_state_parts
        ]

        shape = tf.stack(initial_state_parts).shape
        dtype = dtype_util.base_dtype(tf.stack(initial_state_parts).dtype)

        if initial_covariance is None:
            initial_covariance = 0.001 * tf.eye(
                num_rows=shape[-1], dtype=dtype, batch_shape=[shape[0]]
            )
        else:
            initial_covariance = tf.stack(initial_covariance)

        if mcmc_util.is_list_like(initial_covariance):
            initial_covariance_parts = list(initial_covariance)
        else:
            initial_covariance_parts = [initial_covariance]
        initial_covariance_parts = [
            tf.convert_to_tensor(s, name="initial_covariance")
            for s in initial_covariance_parts
        ]
        initial_covariance_scaling = tf.cast(
            tf.repeat([initial_covariance_scaling], repeats=[shape[0]], axis=0),
            dtype=dtype,
        )

        covariance_scaling_reducer = tf.constant(
            covariance_scaling_reducer, dtype=dtype
        )
        covariance_scaling_limiter = tf.constant(
            covariance_scaling_limiter, dtype=dtype
        )
        target_accept_ratio = tf.constant(target_accept_ratio, dtype=dtype)

        self._running_covar = stats.RunningCovariance(
            shape=(1, shape[-1]), dtype=dtype, event_ndims=1
        )
        self._accum_covar = self._running_covar.initialize()

        self._covariance = initial_covariance
        self._covariance_scaling = initial_covariance_scaling
        self._covariance_scaling_reducer = covariance_scaling_reducer
        self._covariance_scaling_limiter = covariance_scaling_limiter
        self._covariance_burnin = covariance_burnin
        self._covariance_update_interval = covariance_update_interval

        name = mcmc_util.make_name(name, "CovarianceStepAdaptation", "")
        seed_stream = SeedStream(seed, salt="CovarianceStepAdaptation")

        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=initial_state,
            covariance=initial_covariance,
            covariance_scaling=initial_covariance_scaling,
            covariance_scaling_reducer=covariance_scaling_reducer,
            covariance_scaling_limiter=covariance_scaling_limiter,
            covariance_burnin=covariance_burnin,
            covariance_update_interval=covariance_update_interval,
            target_accept_ratio=target_accept_ratio,
            pu=pu,
            fixed_variance=fixed_variance,
            seed=seed,
            extra_setter_fn=rwm_extra_setter_fn,
            results_getter_fn=rwm_results_getter_fn,
            name=name,
        )
        self._impl = metropolis_hastings.MetropolisHastings(
            inner_kernel=UncalibratedMVNRandomWalk(
                target_log_prob_fn=target_log_prob_fn,
                covariance=initial_covariance_parts,
                pu=pu,
                fixed_variance=fixed_variance,
                seed=seed_stream,
                name=name,
            ),
            name=name,
        )

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def initial_state(self):
        return self._parameters["initial_state"]

    @property
    def covariance(self):
        return self._parameters["covariance"]

    @property
    def covariance_scaling(self):
        return self._parameters["covariance_scaling"]

    @property
    def covariance_scaling_reducer(self):
        return self._parameters["covariance_scaling_reducer"]

    @property
    def covariance_scaling_limiter(self):
        return self._parameters["covariance_scaling_limiter"]

    @property
    def covariance_burnin(self):
        return self._parameters["covariance_burnin"]

    @property
    def covariance_update_interval(self):
        return self._parameters["covariance_update_interval"]

    @property
    def target_accept_ratio(self):
        return self._parameters["target_accept_ratio"]

    @property
    def pu(self):
        return self._parameters["pu"]

    @property
    def fixed_variance(self):
        return self._parameters["fixed_variance"]

    @property
    def seed(self):
        return self._parameters["seed"]

    def extra_setter_fn(
        self,
        kernel_results,
        num_steps,
        covariance_scaling,
        covariance,
        running_covariance,
    ):
        return self._parameters["extra_setter_fn"](
            kernel_results,
            num_steps,
            covariance_scaling,
            covariance,
            running_covariance,
        )

    def results_getter_fn(self, kernel_results):
        return self._parameters["results_getter_fn"](kernel_results)

    @property
    def name(self):
        return self._parameters["name"]

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @property
    def running_covar(self):
        return self._running_covar

    @property
    def is_calibrated(self):
        return True

    def one_step(self, current_state, previous_kernel_results, seed=None):
        with tf.name_scope(
            mcmc_util.make_name(self.name, "CovarianceStepAdaptation", "one_step")
        ):
            with tf.name_scope("initialize"):
                if mcmc_util.is_list_like(current_state):
                    current_state_parts = list(current_state)
                else:
                    current_state_parts = [current_state]
                current_state_parts = [
                    tf.convert_to_tensor(s, name="current_state")
                    for s in current_state_parts
                ]

            def update_covariance():
                return self.running_covar.finalize(current_accum_covar, ddof=1)
                # return tf.squeeze(self.running_covar.finalize(current_accum_covar, ddof=1))

            def update_covariance_scaling():
                log_accept_ratio = self.results_getter_fn(
                    previous_kernel_results
                ).log_accept_ratio
                dtype = dtype_util.base_dtype(previous_covariance_scaling.dtype)
                return previous_covariance_scaling * tf.math.exp(
                    tf.math.sign(
                        log_accept_ratio - tf.math.log(self.target_accept_ratio)
                    )
                    * tf.math.minimum(
                        self.covariance_scaling_limiter,
                        tf.cast(num_steps, dtype=dtype)
                        ** (-self.covariance_scaling_reducer),
                    )
                )

            # Note 'covariance_scaling' and 'accum_covar' are updated every step but 'covariance'
            # is updated every 'covariance_update_interval' steps (beyond 'covariance_burning').
            num_steps = self.results_getter_fn(previous_kernel_results).extra.num_steps
            # for parallel processing efficiency use gather() rather than cond()?
            previous_is_adaptive = tf.gather(
                [
                    self.results_getter_fn(
                        previous_kernel_results
                    ).proposed_results.is_adaptive,
                    self.results_getter_fn(
                        previous_kernel_results
                    ).accepted_results.is_adaptive,
                ],
                tf.cast(
                    self.results_getter_fn(previous_kernel_results).is_accepted,
                    dtype=tf.dtypes.int32,
                ),
            )
            previous_covariance_scaling = self.results_getter_fn(
                previous_kernel_results
            ).extra.covariance_scaling
            current_covariance_scaling = tf.gather(
                tf.stack(
                    [previous_covariance_scaling, update_covariance_scaling()], axis=-1
                ),
                previous_is_adaptive,
                batch_dims=1,
                axis=1,
            )
            previous_accum_covar = self.results_getter_fn(
                previous_kernel_results
            ).extra.running_covariance
            current_accum_covar = self.running_covar.update(
                state=previous_accum_covar, new_sample=current_state_parts
            )
            mod = tf.math.mod(num_steps, self.covariance_update_interval)
            previous_covariance = self.results_getter_fn(
                previous_kernel_results
            ).extra.covariance
            current_covariance = tf.gather(
                [previous_covariance, update_covariance()],
                tf.cast(
                    tf.math.logical_and(num_steps >= self.covariance_burnin, mod == 0),
                    dtype=tf.dtypes.int32,
                ),
            )

            ### for testing
            # current_covariance_scaling = self.results_getter_fn(previous_kernel_results).extra.covariance_scaling
            # current_accum_covar = self.results_getter_fn(previous_kernel_results).extra.running_covariance
            # current_covariance = self.results_getter_fn(previous_kernel_results).extra.covariance
            ###

            current_scaled_covariance = tf.squeeze(
                tf.expand_dims(current_covariance_scaling, axis=1)
                * tf.stack([current_covariance]),
                axis=0,
            )
            current_scaled_covariance = tf.unstack(current_scaled_covariance)

            if mcmc_util.is_list_like(current_scaled_covariance):
                current_scaled_covariance_parts = list(current_scaled_covariance)
            else:
                current_scaled_covariance_parts = [current_scaled_covariance]
            current_scaled_covariance_parts = [
                tf.convert_to_tensor(s, name="current_scaled_covariance")
                for s in current_scaled_covariance_parts
            ]

            self._impl = metropolis_hastings.MetropolisHastings(
                inner_kernel=UncalibratedMVNRandomWalk(
                    target_log_prob_fn=self.target_log_prob_fn,
                    covariance=current_scaled_covariance_parts,
                    pu=self.pu,
                    fixed_variance=self.fixed_variance,
                    seed=self.seed,
                    name=self.name,
                ),
                name=self.name,
            )
            new_state, new_inner_results = self._impl.one_step(
                current_state, previous_kernel_results
            )
            new_inner_results = self.extra_setter_fn(
                new_inner_results,
                num_steps + 1,
                tf.squeeze(current_covariance_scaling, axis=1),
                current_covariance,
                current_accum_covar,
            )
            return [new_state, new_inner_results]

    def bootstrap_results(self, init_state):
        """Creates initial `state`."""
        with tf.name_scope(
            mcmc_util.make_name(
                self.name, "CovarianceStepAdaptation", "bootstrap_results"
            )
        ):
            inner_results = self._impl.bootstrap_results(init_state)
            return self.extra_setter_fn(
                inner_results,
                0,
                self.covariance_scaling / self.covariance.shape[0],
                self.covariance,
                self._accum_covar,
            )


class UncalibratedMVNRandomWalk(kernel_base.TransitionKernel):
    """Generate Multivariate Normal Random Walk proposal for the Metropolis Hastings algorithm.
  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use
  `tfp.mcmc.MetropolisHastings(tfp.mcmc.UncalibratedMVNRandomWalk(...))`.
  For more details on `UncalibratedMVNRandomWalk`, see
  `CovarianceStepAdaptation`.
  """

    def __init__(
        self,
        target_log_prob_fn,
        covariance,
        new_state_fn=None,
        pu=0.95,
        fixed_variance=0.01,
        seed=None,
        name=None,
    ):
        """Initializes this transition kernel.
    
    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      covariance: Python `list` of `Tensor`s representing the covariance 
        of the proposal.
      new_state_fn: Python callable which takes a list of state parts and a
        seed; returns a same-type `list` of `Tensor`s, each being a perturbation
        of the input state parts. The perturbation distribution is assumed to be
        a symmetric distribution centered at the input state part.
        Default value `None` is mapped to `random_walk_mvnorm_fn()`.
      pu: Python floating point number, bounded between 0.0 and 1.0, representing 
        the bounded convergence parameter. See `random_walk_mvnorm_fn` for 
        further details.
        Default value: 0.95.
      fixed_variance: Python floating point number representing the variance of
        the fixed proposal distribution. See `random_walk_mvnorm_fn` for 
        further details.
        Default value: 0.01.
      seed: Python integer to seed the random number generator.
        Default value: `None`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None`.

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if `pu` is less than 0.0 or greater than 1.0.
      ValueError: if `fixed_variance` is less than 0.0.
    """
        if mcmc_util.is_list_like(covariance):
            covariance_parts = list(covariance)
        else:
            covariance_parts = [covariance]
        covariance_parts = [
            tf.convert_to_tensor(s, name="covariance") for s in covariance_parts
        ]
        if new_state_fn is None:
            new_state_fn = random_walk_mvnorm_fn(
                covariance=covariance,
                pu=pu,
                fixed_variance=fixed_variance,
                name=mcmc_util.make_name(
                    name, "UncalibratedMVNRandomWalk", "random_walk_mvnorm_fn"
                ),
            )
        if pu < 0.0 or pu > 1.0:
            raise ValueError("`{}` must be a `float` between 0.0 and 1.0.".format("pu"))
        if fixed_variance < 0.0:
            raise ValueError(
                "`{}` must be a `float` greater than 0.0.".format("fixed_variance")
            )
        self._target_log_prob_fn = target_log_prob_fn
        self._covariance = covariance
        self._pu = pu
        self._fixed_variance = fixed_variance
        self._seed_stream = SeedStream(seed, salt="UncalibratedMVNRandomWalk")
        self._name = name
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            covariance=covariance,
            new_state_fn=new_state_fn,
            pu=pu,
            fixed_variance=fixed_variance,
            seed=seed,
            name=name,
        )

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def covariance(self):
        return self._parameters["covariance"]

    @property
    def new_state_fn(self):
        return self._parameters["new_state_fn"]

    @property
    def pu(self):
        return self._parameters["pu"]

    @property
    def fixed_variance(self):
        return self._parameters["fixed_variance"]

    @property
    def seed(self):
        return self._parameters["seed"]

    @property
    def name(self):
        return self._parameters["name"]

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @property
    def is_calibrated(self):
        return False

    def one_step(self, current_state, previous_kernel_results):
        with tf.name_scope(
            mcmc_util.make_name(self.name, "UncalibratedMVNRandomWalk", "one_step")
        ):
            with tf.name_scope("initialize"):
                if mcmc_util.is_list_like(current_state):
                    current_state_parts = list(current_state)
                else:
                    current_state_parts = [current_state]
                current_state_parts = [
                    tf.convert_to_tensor(s, name="current_state")
                    for s in current_state_parts
                ]

            self._new_state_fn = random_walk_mvnorm_fn(
                covariance=self.covariance,
                pu=self.pu,
                fixed_variance=self.fixed_variance,
                name=mcmc_util.make_name(
                    self.name, "UncalibratedMVNRandomWalk", "random_walk_mvnorm_fn"
                ),
            )

            is_adaptive, next_state_parts = self.new_state_fn(
                current_state_parts, self._seed_stream()
            )  # pylint: disable=not-callable
            # Compute `target_log_prob` so its available to MetropolisHastings.
            next_target_log_prob = self.target_log_prob_fn(
                *next_state_parts
            )  # pylint: disable=not-callable

            def maybe_flatten(x):
                return x if mcmc_util.is_list_like(current_state) else x[0]

            return [
                maybe_flatten(next_state_parts),
                UncalibratedMVNRandomWalkResults(
                    log_acceptance_correction=tf.zeros_like(next_target_log_prob),
                    target_log_prob=next_target_log_prob,
                    is_adaptive=tf.cast(is_adaptive, dtype=tf.dtypes.int32),
                ),
            ]

    def bootstrap_results(self, init_state):
        with tf.name_scope(
            mcmc_util.make_name(
                self.name, "UncalibratedMVNRandomWalk", "bootstrap_results"
            )
        ):
            if not mcmc_util.is_list_like(init_state):
                init_state = [init_state]
            init_state = [tf.convert_to_tensor(x) for x in init_state]
            init_target_log_prob = self.target_log_prob_fn(
                *init_state
            )  # pylint:disable=not-callable
            init_is_adaptive = tf.expand_dims(
                tf.zeros([tf.stack(init_state).shape[0]], dtype=tf.dtypes.int32), axis=1
            )
            return UncalibratedMVNRandomWalkResults(
                log_acceptance_correction=tf.zeros_like(init_target_log_prob),
                target_log_prob=init_target_log_prob,
                is_adaptive=init_is_adaptive,
            )
