from mem.cacti_simulation import CactiSimulation
from typing import Dict

## Description missing
class MemoryInstance:
    ## The class constructor
    # Collect all the basic information of a physical memory module.
    # @param mem_config: configuration of memory
    # @param latency: memory access latency (unit: number of cycles).
    # @param r_cost/w_cost: memory unit data access energy in (pJ).
    # @param area: memory area (unit can be whatever user-defined unit).
    # @param min_r_granularity (int): The minimal number of bits than can be read in a clock cycle (can be a less than rw_bw)
    # @param min_w_granularity (int): The minimal number of bits that can be written in a clock cycle (can be less than w_bw)
    # @param mem_type (str): The type of memory. Used for CACTI cost extraction.
    # @param get_cost_from_cacti (bool): Automatically extract the read cost, write cost and area using CACTI.
    def __init__(
        self,
        mem_config: Dict,
        r_cost: float = 0,
        w_cost: float = 0,
        latency: float = 1,
        min_r_granularity=None,
        min_w_granularity=None,
        get_cost_from_cacti: bool = True,
    ):
        if get_cost_from_cacti:
            # Size must be a multiple of 8 when using CACTI
            assert (
                mem_config['size'] % 8 == 0
            ), "Memory size must be a multiple of 8 when automatically extracting costs using CACTI."

            cacti_simulation = CactiSimulation(mem_config)
            mem_config = cacti_simulation.run_cacti()

            self.r_cost = mem_config['r_cost']
            self.w_cost = mem_config['w_cost']
            self.area = mem_config['area']
            self.latency = round(mem_config['latency'], 3)
        else:
            self.r_cost = r_cost
            self.w_cost = w_cost
            self.area = 0
            self.latency = latency

        self.size = mem_config['size']
        self.bank = mem_config['bank_count']
        self.rw_bw = mem_config['rw_bw']
        self.r_port = mem_config['r_port']
        self.w_port = mem_config['w_port']
        self.rw_port = mem_config['rw_port']

        if not min_r_granularity:
            self.r_bw_min = mem_config['rw_bw']
            self.r_cost_min = self.r_cost
        else:
            self.r_bw_min = min_r_granularity
            self.r_cost_min = self.r_cost / (self.rw_bw / self.r_bw_min)

        if not min_w_granularity:
            self.w_bw_min = mem_config['rw_bw']
            self.w_cost_min = self.w_cost
        else:
            self.w_bw_min = min_w_granularity
            self.w_cost_min = self.w_cost / (self.rw_bw / self.w_bw_min)
    
    def get_cacti_cost(self):
        cost = {}
        cost['r_cost'] = self.r_cost 
        cost['w_cost'] = self.w_cost 
        cost['area'] = self.area 
        cost['latency'] = self.latency 
        return cost
    
    ## JSON Representation of this class to save it to a json file.
    def __jsonrepr__(self):
        return self.__dict__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MemoryInstance) and self.__dict__ == other.__dict__

    def __hash__(self):
        return id(self)  # unique for every object within its lifetime

    def __str__(self):
        return f"MemoryInstance({self.name})"

    def __repr__(self):
        return str(self)
