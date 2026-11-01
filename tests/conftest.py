import pytest

from tscm.model import Config, GiB
from tscm.simulate import DEFAULT_MACHINE, example_job, example_params, simulate_runs


def profiling_configs(s_target=200 * GiB, c_e=4):
    """A Phase A + Phase B shaped design: small inputs, shrunk memory, several machine counts."""
    base = dict(machine=DEFAULT_MACHINE, c_e=c_e, e=2, p={"k": 8, "max_iter": 10}, d={"skew": 0.1})
    cfgs = []
    for fraction in (0.005, 0.01, 0.02):
        cfgs.append(Config(s=fraction * s_target, m=1, M_e=48 * GiB, **base))
    for M_e in (3 * GiB, 1.7 * GiB):
        cfgs.append(Config(s=0.02 * s_target, m=1, M_e=M_e, **base))
    for fraction in (0.01, 0.02, 0.04, 0.08):
        for m in (2, 8):
            cfgs.append(Config(s=fraction * s_target, m=m, M_e=24 * GiB, **base))
    return cfgs


@pytest.fixture(scope="session")
def job():
    return example_job()


@pytest.fixture(scope="session")
def truth(job):
    return example_params(job)


@pytest.fixture(scope="session")
def tables(job, truth):
    return simulate_runs(job, truth, profiling_configs(), seed=1)
