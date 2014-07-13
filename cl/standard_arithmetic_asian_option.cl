__kernel void standard_arithmetic_asian_option(__global float* num1, __global float* arith_asian_payoff,float N, float K, float S0, float sigma_sqrt, float drift, float exp_RT,float option_type)
{
    int i = get_global_id(0);
    float former = S0;
    float arith_asian_all = 0.0;
    for(int j=0;j<(int)N;j++){
        float rand = num1[i*(int)N+j];
        float growth_factor = drift * exp(sigma_sqrt * rand);
        former = former * growth_factor;
        arith_asian_all = arith_asian_all + former;
    }
    float arith_asian_mean = arith_asian_all / N;

    if isequal(option_type, 1.0){
        arith_asian_payoff[i] = exp_RT * fmax(arith_asian_mean - K, 0);
    }
    else if isequal(option_type, 2.0){
        arith_asian_payoff[i] = exp_RT * fmax(K - arith_asian_mean, 0);
    }

}