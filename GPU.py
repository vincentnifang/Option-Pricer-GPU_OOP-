__author__ = 'vincent'
import numpy, math, util
import pyopencl as cl
import Quasi_Monte_Carlo as quasi


class CL:
    def __init__(self, kernelargs):
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        self.cntxt = cl.Context(devices=my_gpu_devices)
        # cntxt = cl.create_some_context()
        #now create a command queue in the context
        self.queue = cl.CommandQueue(self.cntxt)
        self.mf = cl.mem_flags

        self.kernelargs = kernelargs


    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        # print fstr
        #create the program
        self.program = cl.Program(self.cntxt, fstr).build()

    def popCorn(self):
        pass

    def execute(self):
        pass

    def ret(self):
        pass

class Option(CL):
    def __init__(self, path_num, Quasi, kernelargs):
        CL.__init__(self, kernelargs)
        self.path_num = path_num
        self.Quasi = Quasi


class BasketOption(Option):
    def popCorn(self):
        #initialize client side (CPU) arrays

        if self.Quasi == True:
            self.rand1 = numpy.array(quasi.GPU_quasi_normal_random(int(self.path_num), 2.0), dtype=numpy.float32)
            self.rand2 = numpy.array(quasi.GPU_quasi_normal_random(int(self.path_num), 2.0), dtype=numpy.float32)
        else:
            self.rand1 = numpy.array(numpy.random.normal(0, 1, (self.path_num, 1)), dtype=numpy.float32)
            self.rand2 = numpy.array(numpy.random.normal(0, 1, (self.path_num, 1)), dtype=numpy.float32)

        self.arith_basket_payoff = numpy.empty(self.rand1.shape, dtype=numpy.float32)
        # create the buffers to hold the values of the input
        self.rand1_buf = cl.Buffer(self.cntxt, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.rand1)
        self.rand2_buf = cl.Buffer(self.cntxt, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.rand2)
        # create output buffer
        self.arith_basket_payoff_buf = cl.Buffer(self.cntxt, self.mf.WRITE_ONLY, self.arith_basket_payoff.nbytes)

    def execute(self):

        # Kernel is now launched
        launch = self.program.standard_arithmetic_basket_option(self.queue, self.rand1.shape, None, self.rand1_buf,
                                                                self.rand2_buf, self.arith_basket_payoff_buf,
                                                                *(self.kernelargs))
        # wait till the process completes
        launch.wait()
        cl.enqueue_read_buffer(self.queue, self.arith_basket_payoff_buf, self.arith_basket_payoff).wait()

    def ret(self):
        p_mean = numpy.mean(self.arith_basket_payoff)
        p_std = numpy.std(self.arith_basket_payoff)
        p_confmc = (p_mean - 1.96 * p_std / math.sqrt(self.path_num), p_mean + 1.96 * p_std / math.sqrt(self.path_num))
        return p_mean, p_std, p_confmc


class BasketOptionWithControlVariate(BasketOption):
    def __init__(self, path_num, Quasi, geo_K, kernelargs):
        BasketOption.__init__(self, path_num, Quasi, kernelargs)
        self.geo_K = geo_K

    def popCorn(self):
        BasketOption.popCorn(self)
        self.geo_basket_payoff = numpy.empty(self.rand1.shape, dtype=numpy.float32)
        self.geo_basket_payoff_buf = cl.Buffer(self.cntxt, self.mf.WRITE_ONLY, self.geo_basket_payoff.nbytes)

    def execute(self):
        # Kernel is now launched
        launch = self.program.geo_mean_arithmetic_basket_option(self.queue, self.rand1.shape, None, self.rand1_buf,
                                                                self.rand2_buf, self.arith_basket_payoff_buf,
                                                                self.geo_basket_payoff_buf, *(self.kernelargs))
        # wait till the process completes
        launch.wait()
        cl.enqueue_read_buffer(self.queue, self.arith_basket_payoff_buf, self.arith_basket_payoff).wait()
        cl.enqueue_read_buffer(self.queue, self.geo_basket_payoff_buf, self.geo_basket_payoff).wait()

    def ret(self):
        # Control Variate
        covxy = numpy.mean(self.geo_basket_payoff * self.arith_basket_payoff) - numpy.mean(
            self.arith_basket_payoff) * numpy.mean(self.geo_basket_payoff)
        theta = covxy / numpy.var(self.geo_basket_payoff)

        # Control Variate Version
        geo = util.geometric_basket_option(S1, S2, V1, V2, R, T, self.geo_K, rou, option_type)
        z = self.arith_basket_payoff + theta * (geo - self.geo_basket_payoff)
        # z = [x + y for x, y in zip(arith_basket_payoff, map(lambda x: theta * (geo - x), geo_basket_payoff))]
        z_mean = numpy.mean(z)
        z_std = numpy.std(z)
        z_confmc = (z_mean - 1.96 * z_std / math.sqrt(path_num), z_mean + 1.96 * z_std / math.sqrt(path_num))
        return z_mean, z_std, z_confmc


class AsianOption(Option):
    def __init__(self, path_num, Quasi, N, kernelargs):
        Option.__init__(self, path_num, Quasi, kernelargs)
        self.N = N

    def popCorn(self):
        #initialize client side (CPU) arrays

        if self.Quasi == True:
            # rand1 = numpy.array(quasi.quasi_normal_random(int(path_num * N), 2.0), dtype=numpy.float32)
            self.rand1 = numpy.array(quasi.GPU_quasi_normal_random(int(path_num * self.N), 2.0), dtype=numpy.float32)
        else:
            self.rand1 = numpy.array(numpy.random.normal(0, 1, (path_num, N)), dtype=numpy.float32)

        self.arith_asian_payoff = numpy.empty((path_num, 1), dtype=numpy.float32)
        # create the buffers to hold the values of the input
        self.rand1_buf = cl.Buffer(self.cntxt, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.rand1)
        # create output buffer
        self.arith_asian_payoff_buf = cl.Buffer(self.cntxt, self.mf.WRITE_ONLY, self.arith_asian_payoff.nbytes)


    def execute(self):

        # Kernel is now launched

        launch = self.program.standard_arithmetic_asian_option(self.queue, (path_num, 1), None, self.rand1_buf,
                                                               self.arith_asian_payoff_buf, *(self.kernelargs))
        # wait till the process completes
        launch.wait()
        cl.enqueue_read_buffer(self.queue, self.arith_asian_payoff_buf, self.arith_asian_payoff).wait()


    def ret(self):
        # print the output
        p_mean = numpy.mean(self.arith_asian_payoff)
        p_std = numpy.std(self.arith_asian_payoff)
        p_confmc = (p_mean - 1.96 * p_std / math.sqrt(path_num), p_mean + 1.96 * p_std / math.sqrt(path_num))
        return p_mean, p_std, p_confmc


class AsianOptionWithControlVariate(AsianOption):
    def __init__(self, path_num, Quasi, N, geo_K, kernelargs):
        AsianOption.__init__(self, path_num, Quasi, N, kernelargs)
        self.geo_K = geo_K

    def popCorn(self):
        AsianOption.popCorn(self)
        self.geo_payoff = numpy.empty((path_num, 1), dtype=numpy.float32)
        self.geo_payoff_buf = cl.Buffer(self.cntxt, self.mf.WRITE_ONLY, self.geo_payoff.nbytes)

    def execute(self):
        launch = self.program.geo_mean_arithmetic_asian_option(self.queue, (path_num, 1), None, self.rand1_buf,
                                                               self.arith_asian_payoff_buf, self.geo_payoff_buf,
                                                               *(self.kernelargs))
        # wait till the process completes
        launch.wait()
        cl.enqueue_read_buffer(self.queue, self.arith_asian_payoff_buf, self.arith_asian_payoff).wait()
        cl.enqueue_read_buffer(self.queue, self.geo_payoff_buf, self.geo_payoff).wait()

    def ret(self):
        covxy = numpy.mean(self.geo_payoff * self.arith_asian_payoff) - numpy.mean(
            self.arith_asian_payoff) * numpy.mean(self.geo_payoff)
        theta = covxy / numpy.var(self.geo_payoff)

        # Control Variate Version
        geo = util.geometric_asian_option(self.geo_K, T, R, V, S0, self.N, option_type)
        # z = [x + y for x, y in zip(arith_payoff, map(lambda x: theta * (geo - x), geo_payoff))]
        z = self.arith_asian_payoff + theta * (geo - self.geo_payoff)
        z_mean = numpy.mean(z)
        z_std = numpy.std(z)
        z_confmc = (z_mean - 1.96 * z_std / math.sqrt(path_num), z_mean + 1.96 * z_std / math.sqrt(path_num))
        return z_mean, z_std, z_confmc


def Test(obj, code):
    obj.loadProgram(code)
    obj.popCorn()
    obj.execute()
    print obj.ret()


def format(f):
    return "%.4f" % f


if __name__ == "__main__":
    path_num = 10000
    Quasi = False

    S = S0 = S1 = S2 = 100.0
    T = 3.0
    R = 0.05
    V = V1 = V2 = 0.3
    geo_K = K = 100.0
    n = 50.0
    rou = 0.5
    m = 10000
    N = 50

    option_type = 1.0

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

    example = BasketOption(path_num, Quasi, kernelargs)
    code = "cl/standard_arithmetic_basket_option.cl"
    Test(example, code)

    S = S0 = S1 = S2 = 100.0
    T = 3.0
    R = 0.05
    V = V1 = V2 = 0.3
    geo_K = K = 100.0
    n = 50.0
    rou = 0.5
    m = 10000
    N = 50

    option_type = 1.0

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

    example = BasketOptionWithControlVariate(path_num, Quasi, geo_K, kernelargs)
    code = "cl/geo_mean_arithmetic_basket_option.cl"
    Test(example, code)

    S = S0 = S1 = S2 = 100.0
    T = 3.0
    R = 0.05
    V = V1 = V2 = 0.3
    geo_K = K = 100.0
    n = 50.0
    rou = 0.5
    m = 10000
    N = 50

    option_type = 1.0

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

    example = AsianOption(path_num, Quasi, N, kernelargs)
    code = "cl/standard_arithmetic_asian_option.cl"
    Test(example, code)

    S = S0 = S1 = S2 = 100.0
    T = 3.0
    R = 0.05
    V = V1 = V2 = 0.3
    geo_K = K = 100.0
    n = 50.0
    rou = 0.5
    m = 10000
    N = 50

    N = numpy.float32(N)
    K = numpy.float32(K)
    geo_K = numpy.float32(geo_K)
    S0 = numpy.float32(S0)
    sigma_sqrt = numpy.float32(sigma_sqrt)
    drift = numpy.float32(drift)
    exp_RT = numpy.float32(exp_RT)
    option_type = numpy.float32(option_type)

    kernelargs = (N, K, geo_K, S0, sigma_sqrt, drift, exp_RT, option_type)

    example = AsianOptionWithControlVariate(path_num, Quasi, N, geo_K, kernelargs)
    code = "cl/geo_mean_arithmetic_asian_option.cl"
    Test(example, code)




