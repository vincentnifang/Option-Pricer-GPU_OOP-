__kernel void european_option(__global float* num1, __global float* european_payoff,float N, float K, float S0, float sigma_sqrt, float drift, float exp_RT,float option_type)
{
    int i = get_global_id(0);
    float former = S0;
    for(int j=0;j<(int)N;j++){
        float rand = num1[i*(int)N+j];
        former = former * drift * exp(sigma_sqrt * rand);
    }
    float european = former;
    if isequal(option_type, 1.0){
        european_payoff[i] = exp_RT * fmax(european - K, 0);
    }
    else if isequal(option_type, 2.0){
        european_payoff[i] = exp_RT * fmax(K - european, 0);
    }

}