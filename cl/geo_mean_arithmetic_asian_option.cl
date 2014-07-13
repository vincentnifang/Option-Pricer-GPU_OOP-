__kernel void geo_mean_arithmetic_asian_option(__global float* num1, __global float* arith_payoff,__global float* geo_payoff, float N, float K,float geo_K, float S0, float sigma_sqrt, float drift, float exp_RT,float option_type)
{
    int i = get_global_id(0);
    float former = S0;
    float arith_asian_all = 0.0;
    float geo_asian_all = 0.0;
    for(int j=0;j<(int)N;j++){
        float rand = num1[i*(int)N+j];
        float growth_factor = drift * exp(sigma_sqrt * rand);
        former = former * growth_factor;
        arith_asian_all = arith_asian_all + former;
        geo_asian_all = geo_asian_all + log(former);
    }
    float arith_asian_mean = arith_asian_all / N;
    float geo_asian_mean = exp(geo_asian_all / N);

    if isequal(option_type, 1.0){
        arith_payoff[i] = exp_RT * fmax(arith_asian_mean - K, 0);
        geo_payoff[i] = exp_RT * fmax(geo_asian_mean - geo_K, 0);
    }
    else if isequal(option_type, 2.0){
        arith_payoff[i] = exp_RT * fmax(K - arith_asian_mean, 0);
        geo_payoff[i] = exp_RT * fmax(geo_K - geo_asian_mean, 0);
    }

}