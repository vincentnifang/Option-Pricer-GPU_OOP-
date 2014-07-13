__author__ = 'vincent'
import math
from scipy import stats




def geometric_basket_option(S1, S2, V1, V2, R, T, K, rou, option_type):
    bg_0 = math.sqrt(S1 * S2)
    sigma_bg = math.sqrt(V1 * V1 + V1 * V2 * rou + V2 * V1 * rou + V2 * V2) / 2
    miu_bg = R - 0.5 * (V1 * V1 + V2 * V2) / 2 + 0.5 * sigma_bg * sigma_bg

    d1 = (math.log(bg_0 / K) + (miu_bg + 0.5 * sigma_bg * sigma_bg) * T) / (sigma_bg * math.sqrt(T))
    d2 = d1 - sigma_bg * math.sqrt(T)

    if option_type == 1.0:
        return math.exp(-R * T) * (bg_0 * math.exp(miu_bg * T) * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    elif option_type == 2.0:
        return math.exp(-R * T) * (K * stats.norm.cdf(-d2) - bg_0 * math.exp(miu_bg * T) * stats.norm.cdf(-d1))
    return None


def _get_geometric_sigma(V, N):
    return V * math.sqrt((N + 1) * (2 * N + 1) / (6 * N * N))


def _get_geometric_miu(R, V, N, sigma):
    return (R - 0.5 * V * V) * (N + 1) / (2 * N) + 0.5 * sigma * sigma


def geometric_asian_option(K, T, R, V, S0, N, option_type):
    sigma = _get_geometric_sigma(V, N)
    miu = _get_geometric_miu(R, V, N, sigma)
    d1 = (math.log(S0 / K) + (miu + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 1.0:
        return math.exp(-R * T) * (S0 * math.exp(miu * T) * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    elif option_type == 2.0:
        return math.exp(-R * T) * (K * stats.norm.cdf(-d2) - S0 * math.exp(miu * T) * stats.norm.cdf(-d1))
    return None
