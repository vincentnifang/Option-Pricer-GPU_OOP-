__author__ = 'vincent'
import numpy, math, util
import pdb
from Option import BasketOption,BasketOptionWithControlVariate,AsianOption,AsianOptionWithControlVariate
from Option import EuropeanOption

STANDARD = 'Standard'
GEO_MEAN = 'Geometric mean Asian'
GEO_MEAN_STRIKE = 'Geometric mean Asian with adjusted strike'


def GPU_arithmetic_basket_option(S1, S2, V1, V2, R, T, K, geo_K, rou, option_type, path_num=10000,
                                 control_variate='Standard', Quasi=True):
    if control_variate == STANDARD:

        S1 = numpy.float32(S1)
        S2 = numpy.float32(S2)
        V1 = numpy.float32(V1)
        V2 = numpy.float32(V2)
        R = numpy.float32(R)
        K = numpy.float32(K)
        T = numpy.float32(T)
        rou = numpy.float32(rou)
        option_type = numpy.float32(option_type)

        kernelargs = (S1, S2, V1, V2, R, K, T, rou, option_type)

        code = "cl/standard_arithmetic_basket_option.cl"
        return BasketOption(path_num, Quasi, kernelargs).cal(code)


    elif control_variate == GEO_MEAN:

        S1 = numpy.float32(S1)
        S2 = numpy.float32(S2)
        V1 = numpy.float32(V1)
        V2 = numpy.float32(V2)
        R = numpy.float32(R)
        K = numpy.float32(K)
        T = numpy.float32(T)
        rou = numpy.float32(rou)
        option_type = numpy.float32(option_type)
        geo_K = numpy.float32(geo_K)

        kernelargs = (S1, S2, V1, V2, R, K, geo_K, T, rou, option_type)

        code = "cl/geo_mean_arithmetic_basket_option.cl"
        return BasketOptionWithControlVariate(path_num, Quasi, geo_K, kernelargs).cal(code)

    elif control_variate == GEO_MEAN_STRIKE:
        # K = K + mean(bgT)-mean(baT)
        # mean(bgT) = e^(mu*T)*bg_0 = e^(mu*T)*S0
        # mean(baT) = 1/n * sum(S(T)) = 1/n * sum(S0*e^(rt))
        bg_0 = math.sqrt(S1 * S2)
        sigma_bg = math.sqrt(V1 * V1 + V1 * V2 * rou + V2 * V1 * rou + V2 * V2) / 2
        miu_bg = R - 0.5 * (V1 * V1 + V2 * V2) / 2 + 0.5 * sigma_bg * sigma_bg

        E_bg = bg_0 * math.exp(miu_bg * T)
        E_ba = (S1 * math.exp(R * T) + S2 * math.exp(R * T)) / 2
        geo_K = K + E_bg - E_ba
        return GPU_arithmetic_basket_option(S1, S2, V1, V2, R, T, K, geo_K, rou, option_type, path_num, GEO_MEAN)


def GPU_arithmetic_asian_option(K, geo_K, T, R, V, S0, N, option_type, path_num=10000, control_variate='Standard',
                                Quasi=True):
    if control_variate == STANDARD:

        dt = T / N
        sigma = V
        drift = math.exp((R - 0.5 * sigma * sigma) * dt)
        sigma_sqrt = sigma * math.sqrt(dt)
        exp_RT = math.exp(-R * T)

        N = numpy.float32(N)
        K = numpy.float32(K)
        S0 = numpy.float32(S0)
        sigma_sqrt = numpy.float32(sigma_sqrt)
        drift = numpy.float32(drift)
        exp_RT = numpy.float32(exp_RT)
        option_type = numpy.float32(option_type)

        kernelargs = (N, K, S0, sigma_sqrt, drift, exp_RT, option_type)

        code = "cl/standard_arithmetic_asian_option.cl"
        return AsianOption(path_num, Quasi, N, kernelargs).cal(code)

    elif control_variate == GEO_MEAN:
        dt = T / N
        sigma = V
        drift = math.exp((R - 0.5 * sigma * sigma) * dt)
        sigma_sqrt = sigma * math.sqrt(dt)
        exp_RT = math.exp(-R * T)

        N = numpy.float32(N)
        K = numpy.float32(K)
        geo_K = numpy.float32(geo_K)
        S0 = numpy.float32(S0)
        sigma_sqrt = numpy.float32(sigma_sqrt)
        drift = numpy.float32(drift)
        exp_RT = numpy.float32(exp_RT)
        option_type = numpy.float32(option_type)

        kernelargs = (N, K, geo_K, S0, sigma_sqrt, drift, exp_RT, option_type)


        code = "cl/geo_mean_arithmetic_asian_option.cl"
        return AsianOptionWithControlVariate(path_num, Quasi, N, geo_K, kernelargs).cal(code)

    elif control_variate == GEO_MEAN_STRIKE:
        # K = K + mean(agT)-mean(aaT)
        # mean(agT) = e^(mu*T)*ag_0 = e^(mu*T)*S0
        # mean(aaT) = 1/n * sum(S(T)) = 1/n * sum(S0*e^(rt))
        sigma = util._get_geometric_sigma(V, N)
        miu = util._get_geometric_miu(R, V, N, sigma)
        E_ag = S0 * math.exp(miu * T)

        dt = T / N
        E_aa = sum([math.exp(R * (i + 1) * dt) for i in xrange(int(N))]) * S0 / N
        geo_K = K + E_ag - E_aa
        return GPU_arithmetic_asian_option(K, geo_K, T, R, V, S0, N, option_type, path_num, GEO_MEAN)


def standardMC_european_option(K, T, R, V, S0, N, option_type, path_num=10000):
    dt = T / N
    sigma = V
    drift = math.exp((R - 0.5 * sigma * sigma) * dt)
    sigma_sqrt = sigma * math.sqrt(dt)
    exp_RT = math.exp(-R * T)
    european_payoff = []
    for i in xrange(path_num):
        former = S0
        for j in xrange(int(N)):
            former = former * drift * math.exp(sigma_sqrt * numpy.random.normal(0, 1))
        european_option = former

        if option_type == '1.0':
            european_payoff_call = exp_RT * max(european_option - K, 0)
            european_payoff.append(european_payoff_call)
        elif option_type == '2.0':
            european_payoff_put = exp_RT * max(K - european_option, 0)
            european_payoff.append(european_payoff_put)

    # Standard Monte Carlo
    p_mean = numpy.mean(european_payoff)
    p_std = numpy.std(european_payoff)
    p_confmc = (p_mean - 1.96 * p_std / math.sqrt(path_num), p_mean + 1.96 * p_std / math.sqrt(path_num))
    return p_mean, p_std, p_confmc

def GPU_european_option(K, T, R, V, S0, N, option_type, path_num=10000, Quasi=True):
    # pdb.set_trace()
    dt = T / N
    sigma = V
    drift = math.exp((R - 0.5 * sigma * sigma) * dt)
    sigma_sqrt = sigma * math.sqrt(dt)
    exp_RT = math.exp(-R * T)


    N = numpy.float32(N)
    K = numpy.float32(K)
    S0 = numpy.float32(S0)
    sigma_sqrt = numpy.float32(sigma_sqrt)
    drift = numpy.float32(drift)
    exp_RT = numpy.float32(exp_RT)
    option_type = numpy.float32(option_type)

    kernelargs = (N, K, S0, sigma_sqrt, drift, exp_RT, option_type)

    code = "cl/european_option.cl"
    return EuropeanOption(path_num, Quasi, N, kernelargs).cal(code)



if __name__ == '__main__':
    #print"S=100,K=100,t=0,T=0.5,v=20%,and r=1%."
    import time

    s = time.time()

    S = S0 = S1 = S2 = 100.0
    T = 3.0
    R = 0.05
    V = V1 = V2 = 0.3
    K = 100.0
    n = 50.0
    rou = 0.5
    m = 10000

    print GPU_arithmetic_asian_option(K, K, T, R, V, S0, n, 1.0, path_num=100, control_variate=STANDARD, Quasi=True)

    import project

    print project.arithmetic_asian_option(K, K, T, R, V, S0, n, 'call', path_num=100, control_variate=STANDARD)

    e = time.time()
    print "use", e - s

    s = time.time()
    print standardMC_european_option(K, T, R, V, S0, n, 'call', path_num=100000)
    e = time.time()
    print "use", e - s

    print project.bs(S0, K, T, V, R, 'call')
    s = time.time()
    print GPU_european_option(K, T, R, V, S0, n, 1.0, path_num=100000, Quasi=True)
    e = time.time()
    print "use", e - s