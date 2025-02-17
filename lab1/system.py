from unit import Unit

class System(object):

    def __init__(self, offchip_mem_bw=900,
                 compute_efficiency=1, memory_efficiency=1, flops=123,
                 frequency=940):

        self.unit = Unit()

        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')  ## Absolute number, Unit = GB/s

        self.compute_efficiency = compute_efficiency
        self.memory_efficiency = memory_efficiency

        # flops : # of floating point operations
        self.flops = self.unit.unit_to_raw(flops, type='C')  ## Absolute number, Unit = TFLOPS
        self.op_per_sec = self.flops/2
        self.frequency = self.unit.unit_to_raw(frequency, type='F')  ## Absolute number, Unit = MHz


    def __str__(self):
        unit = Unit()
        a = f"Accelerator OPS: {unit.raw_to_unit(self.flops,type='C')} TOPS , Freq = {unit.raw_to_unit(self.frequency,type='F')} GHz \n"
        c = f"Off-chip mem BW:{unit.raw_to_unit(self.offchip_mem_bw, type='BW')} GB/s \n"
        return a+c


    def set_offchip_mem_bw(self,offchip_mem_bw):
        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')

    def get_offchip_mem_bw(self):
        return self.unit.raw_to_unit(self.offchip_mem_bw,type='BW')


