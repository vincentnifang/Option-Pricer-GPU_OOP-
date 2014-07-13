__kernel void geo_mean_arithmetic_basket_option(__global float* num1, __global float* num2, __global float* out,__global float* geo, float S1, float S2, float V1, float V2, float R, float K,float geo_K, float T, float rou, float option_type)
{
    int i = get_global_id(0);
    float ran1 = num1[i];
    float ran2 = rou * ran1 + sqrt(1 - rou * rou) * num2[i];
    float a1 = S1 * exp((R - 0.5 * V1 * V1) * T + V1 * sqrt(T) * ran1);
    float a2 = S2 * exp((R - 0.5 * V2 * V2) * T + V2 * sqrt(T) * ran2);
    float arith_basket_mean = (a1 + a2) / 2;

    float geo_basket_mean = sqrt(a1 * a2);

    if isequal(option_type, 1.0){
        float arith_basket_payoff_call = exp(-R * T) * fmax(arith_basket_mean - K, 0);
        out[i] = arith_basket_payoff_call;
        float geo_basket_payoff_call = exp(-R * T) * fmax(geo_basket_mean - geo_K, 0);
        geo[i] = geo_basket_payoff_call;
    }
    else if isequal(option_type, 2.0){
        float arith_basket_payoff_put = exp(-R * T) * fmax(K - arith_basket_mean, 0);
        out[i] = arith_basket_payoff_put;
        float geo_basket_payoff_put = exp(-R * T) * fmax(geo_K - geo_basket_mean, 0);
        geo[i] = geo_basket_payoff_put;
    }

}