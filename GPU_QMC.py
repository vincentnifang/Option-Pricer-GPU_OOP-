__author__ = 'vincent'
from scipy import stats
import pyopencl as cl
import numpy, math, time


def generate_GPU_QMC_random(num):

    # print 1 % 2

    cntxt = cl.create_some_context()
    #now create a command queue in the context
    queue = cl.CommandQueue(cntxt)
    # create some data array to give as input to Kernel and get output
    input = numpy.array(xrange(1,num+1), dtype=numpy.float32)

    # print input
    out = numpy.empty(input.shape, dtype=numpy.float32)
    # create the buffers to hold the values of the input
    input_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY |
                                 cl.mem_flags.COPY_HOST_PTR, hostbuf=input)
    # create output buffer
    out_buf = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, out.nbytes)


    # Kernel Program
    code = """
    __kernel void qpu_qmc(__global float* input, __global float* out)
    {
        int i = get_global_id(0);
        double result = 0.0;
        int base = 2;
        double f = 1 / (double)base;
        int j = input[i];
        while (j > 0){
            result = result + f * (j & 1);
            j = floor((double)j/(double)base);
            f = f / (double)base;
        }

        double a0 = 2.50662823884;
        double b0 = -8.47351093090;
        double a1 = -18.61500062529;
        double b1 = 23.08336743743;
        double a2 = 41.39119773534;
        double b2 = -21.06224101826;
        double a3 = -25.44106049637;
        double b3 = 3.13082909833;
        double c0 = 0.3374754822726147;
        double c5 = 0.0003951896511919;
        double c1 = 0.9761690190917186;
        double c6 = 0.0000321767881768;
        double c2 = 0.1607979714918209;
        double c7 = 0.0000002888167364;
        double c3 = 0.0276438810333863;
        double c8 = 0.0000003960315187;
        double c4 = 0.0038405729373609;

        double u = result;
        double y = u - 0.5;
        double r;
        double x;
        if (fabs(y) < 0.42){
            r = y * y;
            x = y *  (((a3 * r + a2) * r + a1) * r + a0) / ((((b3 * r + b2) * r + b1) * r + b0) * r +1);
            }
        else {
            r = u;
            if (y > 0){
                r = 1 - u;
                }
            r = log(-log(r));
            x = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4+ r * (c5 + r * (c6 + r * (c7 + r * c8)))))));
            if (y < 0){
                x = -x;
                }
            }
        out[i] = x;

    }
    """

    #
    # FUNCTION (index, base)
    #    BEGIN
    #        result = 0;
    #        f = 1 / base;
    #        i = index;
    #        WHILE (i > 0)
    #        BEGIN
    #            result = result + f * (i % base);
    #            i = FLOOR(i / base);
    #            f = f / base;
    #        END
    #        RETURN result;
    #    END


    # build the Kernel
    bld = cl.Program(cntxt, code).build()

    # Kernel is now launched
    launch = bld.qpu_qmc(queue, input.shape, None, input_buf, out_buf)
    # wait till the process completes
    launch.wait()
    cl.enqueue_read_buffer(queue, out_buf, out).wait()
    # print the output

    return out