__author__ = 'vincent'
import pyopencl as cl

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

    def cal(self, filename):
        self.loadProgram(filename)
        self.popCorn()
        self.execute()
        return self.ret()