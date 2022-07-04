import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import phirl.mdp.interfaces as mdp_int
import numpy as np
from typing import TypeVar

_S = TypeVar("_S")
_A = TypeVar("_A")


def sample_reward(params: mdp_int.IMDPParams[_S, _A], distribution: str, dist_params) -> np.ndarray:
    if distribution == "uniform":
        return np.random.uniform(dist_params[0], dist_params[1],size=params.n_states)

    elif distribution == "gaussian":
        return np.random.normal(dist_params[0], dist_params[1], params.n_states)

    else:
        raise NotImplementedError('{} is not implemented.'.format(distribution))


def sample_cluster(n_clusters, seed = 100) -> int:
    
    key = jax.random.PRNGKey(seed)
    dirichlet = dist.Dirichlet(jnp.ones(n_clusters))
    probs = dirichlet.sample(key)
    categorical_sampler = dist.Categorical(probs=probs)
    clusterIdx = categorical_sampler.sample(key)
    """
    dirichlet = np.random.dirichlet(alpha=np.ones(n_clusters))
    clusterIdx = np.random.choice(n_clusters,p=dirichlet)
    """
    
    return clusterIdx


    
