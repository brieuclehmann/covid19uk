"""Functions for chain binomial simulation."""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def approx_expm(rates):
    """Approximates a full Markov transition matrix
    :param rates: un-normalised rate matrix (i.e. diagonal zero)
    :returns: approximation to Markov transition matrix
    """
    total_rates = tf.reduce_sum(rates, axis=-1, keepdims=True)
    prob = 1. - tf.math.exp(-tf.reduce_sum(rates, axis=-1, keepdims=True))
    mt1 = tf.math.multiply_no_nan(rates / total_rates, prob)
    return tf.linalg.set_diag(mt1, 1. - tf.reduce_sum(mt1, axis=-1))


def chain_binomial_propagate(h, time_step, seed=None):
    """Propagates the state of a population according to discrete time dynamics.

    :param h: a hazard rate function returning the non-row-normalised Markov transition rate matrix
              This function should return a tensor of dimension [ns, ns, nc] where ns is the number of
              states, and nc is the number of strata within the population.
    :param time_step: the time step
    :returns : a function that propagate `state[t]` -> `state[t+time_step]`
    """
    def propagate_fn(t, state):
        rate_matrix = h(t, state)
        # Set diagonal to be the negative of the sum of other elements in each row
        markov_transition = approx_expm(rate_matrix*time_step)
        num_states = markov_transition.shape[-1]
        prev_probs = tf.zeros_like(markov_transition[..., :, 0])
        counts = tf.zeros(markov_transition.shape[:-1].as_list() + [0],
                          dtype=markov_transition.dtype)
        total_count = state
        # This for loop is ok because there are (currently) only 4 states (SEIR)
        # and we're only actually creating work for 3 of them. Even for as many
        # as a ~10 states it should probably be fine, just increasing the size
        # of the graph a bit.
        for i in range(num_states - 1):
          probs = markov_transition[..., :, i]
          binom = tfd.Binomial(
              total_count=total_count,
              probs=tf.clip_by_value(probs / (1. - prev_probs), 0., 1.))
          sample = binom.sample(seed=seed)
          counts = tf.concat([counts, sample[..., tf.newaxis]], axis=-1)
          total_count -= sample
          prev_probs += probs

        counts = tf.concat([counts, total_count[..., tf.newaxis]], axis=-1)
        new_state = tf.reduce_sum(counts, axis=-2)
        return counts, new_state
    return propagate_fn


def discrete_markov_simulation(hazard_fn, state, start, end, time_step, seed=None):
    """Simulates from a discrete time Markov state transition model using multinomial sampling
    across rows of the """
    propagate = chain_binomial_propagate(hazard_fn, time_step, seed=seed)
    times = tf.range(start, end, time_step)
    state = tf.convert_to_tensor(state)

    output = tf.TensorArray(state.dtype, size=times.shape[0])

    cond = lambda i, *_: i < times.shape[0]
    def body(i, state, output):
      update, state = propagate(i, state)
      output = output.write(i, update)
      return i + 1, state, output
    _, state, output = tf.while_loop(cond, body, loop_vars=(0, state, output))
    return times, output.stack()


def discrete_markov_log_prob(events, init_state, hazard_fn, time_step):
    """Calculates an unnormalised log_prob function for a discrete time epidemic model.
    :param events: a [n_t, n_c, n_s, n_s] batch of transition events for all times t, metapopulations c,
                   and states s
    :param init_state: a vector of shape [n_c, n_s] the initial state of the epidemic for s states
                       and c metapopulations
    :param hazard_fn: a function that takes a state and returns a matrix of transition rates
    """
    t = tf.range(events.shape[-4])

    logp_parts = tf.TensorArray(dtype=events.dtype, size=t.shape[0])

    def log_prob_t(i, state, accum):
        # Calculate transition rate matrix
        rate_matrix = hazard_fn(i, state)  # Todo ideally this should be batched, not iterated
        markov_transition = approx_expm(rate_matrix*time_step)
        # Calculate event matrix
        event_matrix = tf.linalg.set_diag(events[i], state - tf.reduce_sum(events[i], axis=-1))
        logp = tfd.Multinomial(state, probs=markov_transition).log_prob(event_matrix)
        new_state = tf.reduce_sum(event_matrix, axis=-2)
        return i+1, new_state, accum.write(i, logp)

    def cond(i, _1, _2):
        return i < t.shape[0]

    # Todo This loop tries to avoid batching issues in the rate calculations.  Maybe a bottleneck if
    #   the computations within the rates themselves are trivial.
    _, _, logp_parts = tf.while_loop(cond, log_prob_t, (0, init_state, logp_parts))
    return logp_parts.stack()


def events_to_full_transitions(events, initial_state):
    """Creates a state tensor given matrices of transition events
    and the initial state
    :param events: a tensor of shape [t, c, s, s] for t timepoints, c metapopulations
                   and s states.
    :param initial_state: the initial state matrix of shape [c, s]
    """
    def f(state, events):
        survived = tf.reduce_sum(state, axis=-2) - tf.reduce_sum(events, axis=-1)
        new_state = tf.linalg.set_diag(events, survived)
        return new_state

    return tf.scan(fn=f, elems=events, initializer=tf.linalg.diag(initial_state))

