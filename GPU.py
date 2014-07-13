__author__ = 'vincent'
import numpy,math
import pyopencl as cl
import Quasi_Monte_Carlo as quasi

class CL:
    def __init__(self, path_num, kernelargs):
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        self.cntxt = cl.Context(devices=my_gpu_devices)
        # cntxt = cl.create_some_context()
        #now create a command queue in the context
        self.queue = cl.CommandQueue(self.cntxt)

        self.path_num = path_num
        self.kernelargs = kernelargs

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        print fstr
        #create the program
        self.program = cl.Program(self.cntxt, fstr).build()

    def popCorn(self,Quasi):
        mf = cl.mem_flags

        #initialize client side (CPU) arrays

        if Quasi == True:
            self.rand1 = numpy.array(quasi.GPU_quasi_normal_random(int(self.path_num), 2.0), dtype=numpy.float32)
            self.rand2 = numpy.array(quasi.GPU_quasi_normal_random(int(self.path_num), 2.0), dtype=numpy.float32)
        else:
            self.rand1 = numpy.array(numpy.random.normal(0, 1, (self.path_num, 1)), dtype=numpy.float32)
            self.rand2 = numpy.array(numpy.random.normal(0, 1, (self.path_num, 1)), dtype=numpy.float32)

        self.arith_basket_payoff = numpy.empty(self.rand1.shape, dtype=numpy.float32)
        # create the buffers to hold the values of the input
        self.rand1_buf = cl.Buffer(self.cntxt, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.rand1)
        self.rand2_buf = cl.Buffer(self.cntxt, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.rand2)
        # create output buffer
        self.arith_basket_payoff_buf = cl.Buffer(self.cntxt, mf.WRITE_ONLY, self.arith_basket_payoff.nbytes)


    def execute(self):

        # Kernel is now launched
        launch = self.program.standard_arithmetic_basket_option(self.queue, self.rand1.shape, None, self.rand1_buf, self.rand2_buf,
                                                       self.arith_basket_payoff_buf, *(self.kernelargs))
        # wait till the process completes
        launch.wait()
        cl.enqueue_read_buffer(self.queue, self.arith_basket_payoff_buf, self.arith_basket_payoff).wait()

    def ret(self):
        p_mean = numpy.mean(self.arith_basket_payoff)
        p_std = numpy.std(self.arith_basket_payoff)
        p_confmc = (p_mean - 1.96 * p_std / math.sqrt(path_num), p_mean + 1.96 * p_std / math.sqrt(path_num))
        return p_mean, p_std, p_confmc



if __name__ == "__main__":
    path_num = 10000
    Quasi = True

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



    example = CL(path_num, kernelargs)
    example.loadProgram("cl/standard_arithmetic_basket_option.cl")
    example.popCorn(Quasi)
    example.execute()
    print example.ret()

