import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
from tqdm import tqdm
tfd = tfp.distributions
tf.experimental.numpy.experimental_enable_numpy_behavior()

dtype = np.float32
N = 100

#referenced from wikipedia and stackoverflow
def kl_mvn(m0, S0, m1, S1):
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) 
    quad_term = diff.T @ np.linalg.inv(S1) @ diff 
    D_kl = .5 * (tr_term + det_term + quad_term - N)
    #pinskers
    return np.sqrt(.5*D_kl)

#Implementations for three methods based off tensorflow docs.
def RWMH(d, true_mean, true_cov):
    
    #posterior = tfd.MultivariateNormalDiag(loc = mu_N, scale_diag = sigma_N)
    #define target via cholesky decomposition (explain this)
    L = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean,scale_tril=L)
    num_results = 1000
    num_chains = 100
    init_state = np.ones([num_chains, d], dtype=dtype)

    samples = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target.log_prob),
        num_burnin_steps=0,
        num_steps_between_results=0,
        trace_fn=None,
        seed=2024)

    sample_mean = tf.math.reduce_mean(samples, axis=0)
    mean_sample_mean = tf.math.reduce_mean(sample_mean, axis=0)
    x = tf.squeeze(samples - sample_mean)
    sample_cov = tf.matmul(tf.transpose(x, [1, 2, 0]),
                        tf.transpose(x, [1, 0, 2])) / num_results
    mean_sample_cov = tf.math.reduce_mean(sample_cov, axis=0)

    return true_mean,  true_cov, mean_sample_mean, mean_sample_cov

def MALA(d, true_mean, true_cov):
    L = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=L)
    num_results = 1000
    num_chains = 100
    init_state = np.ones([num_chains, d], dtype=dtype)

    samples = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target.log_prob,
            step_size=.1),
        num_burnin_steps=0,
        num_steps_between_results=0,
        trace_fn=None,
        seed=2024)

    sample_mean = tf.reduce_mean(samples, axis=[0, 1])
    x = (samples - sample_mean)[..., tf.newaxis]
    sample_cov = tf.reduce_mean(
    tf.matmul(x, tf.transpose(x, [0, 1, 3, 2])), [0, 1])

    return true_mean, true_cov, sample_mean, sample_cov

def HMC(d, true_mean, true_cov):
    L = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=L)
    num_results = 1000
    num_chains = 100
    init_state = np.ones([num_chains, d], dtype=dtype)

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target.log_prob,
        num_leapfrog_steps=2,
        step_size=1.),
        num_adaptation_steps=100)

    samples = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=adaptive_hmc,
        num_burnin_steps=0,
        num_steps_between_results=0,
        trace_fn=None,
        seed=2024)
    
    sample_mean = tf.reduce_mean(samples, axis=[0, 1])
    x = (samples - sample_mean)[..., tf.newaxis]
    sample_cov = tf.reduce_mean(
    tf.matmul(x, tf.transpose(x, [0, 1, 3, 2])), [0, 1])

    return true_mean, true_cov, sample_mean, sample_cov


def simulate(d, n_iter):
    mvn = tfd.MultivariateNormalDiag(loc = np.zeros((1,d)), scale_diag = np.eye(d))

    mu = tf.squeeze(tf.linalg.diag_part(mvn.sample(1)))
    likelihood = tfd.MultivariateNormalDiag(loc = mu, scale_diag = (1/N)*np.eye(d))
    #from clt can directly sample xbar (prove this)
    x_bar = likelihood.sample(1)

    true_cov = dtype((1/N+1)*np.eye(d))
    mu_N = dtype((N/N+1)*x_bar*np.eye(d))
    true_mean = tf.squeeze(tf.linalg.diag_part(mu_N))

    rwmh_set = np.zeros(n_iter)
    mala_set = np.zeros(n_iter)
    hmc_set = np.zeros(n_iter)
    for i in tqdm(range(0, n_iter), leave=False):
        #RWMH
        mean_rwmh, cov_rwmh, sample_mean_rwmh, sample_cov_rwmh = RWMH(d,true_mean, true_cov)
        #MALA
        mean_mala, cov_mala, sample_mean_mala, sample_cov_mala = MALA(d, true_mean, true_cov)
        #HMC
        mean_hmc, cov_hmc, sample_mean_hmc, sample_cov_hmc = HMC(d, true_mean, true_cov)
        #bounds
        rwmh_bounds = kl_mvn(mean_rwmh, cov_rwmh, sample_mean_rwmh, sample_cov_rwmh)
        mala_bounds = kl_mvn(mean_mala, cov_mala, sample_mean_mala, sample_cov_mala)
        hmc_bounds = kl_mvn(mean_hmc, cov_hmc, sample_mean_hmc, sample_cov_hmc)
        
        hmc_set[i] = hmc_bounds
        rwmh_set[i] = rwmh_bounds
        mala_set[i] = mala_bounds
    return rwmh_set, mala_set, hmc_set

def gen_plot():
    rwmh_means = np.zeros(10)
    rwmh_lowers = np.zeros(10)
    rwmh_uppers = np.zeros(10)

    mala_means = np.zeros(10)
    mala_lowers = np.zeros(10)
    mala_uppers = np.zeros(10)

    hmc_means = np.zeros(10)
    hmc_lowers = np.zeros(10)
    hmc_uppers = np.zeros(10)

    ind = 0

    dims = range(2,22,2)

    for d in tqdm(range(2,22,2), leave=False):
        rwmh, mala, hmc = simulate(d,10)

        rwmh_means[ind] = np.mean(rwmh,axis=0)
        rwmh_lowers[ind] = rwmh_means[ind] - 1.96*np.std(rwmh)
        rwmh_uppers[ind] = rwmh_means[ind] + 1.96*np.std(rwmh)

        mala_means[ind] = np.mean(mala, axis=0)
        mala_lowers[ind] = mala_means[ind] - 1.96*np.std(mala)
        mala_uppers[ind] = mala_means[ind] + 1.96*np.std(mala)

        hmc_means[ind] = np.mean(hmc, axis=0)
        hmc_lowers[ind] = hmc_means[ind] - 1.96*np.std(hmc)
        hmc_uppers[ind] = hmc_means[ind] + 1.96*np.std(hmc)
        
        ind = ind + 1
        print(ind)

    f = open('objs.pkl','wb')
    pickle.dump([rwmh_means,rwmh_lowers,rwmh_uppers,
                 mala_means,mala_lowers,mala_uppers,
                 hmc_means,hmc_lowers,hmc_uppers,dims],f)
    f.close()

gen_plot()


