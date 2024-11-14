import os, math
from subprocess import PIPE, Popen
from mem.cacti_config import CactiConfig

class CactiSimulation:
    ## Path of current directory
    CACTI_TOP_PATH = os.path.dirname(os.path.realpath(__file__))
    ## Path to cacti program
    CACTI_PROGRAM_PATH = os.path.join(CACTI_TOP_PATH, 'cacti')
    ## Path to store the CACTI config file
    CACTI_CONFIG_PATH = os.path.join(CACTI_TOP_PATH, 'self_gen')
    ## CACTI output file
    CACTI_OUTPUT_FILE = os.path.join(CACTI_CONFIG_PATH, 'cache.cfg.out')

    def __init__(self, mem_config):
        self._check_valid_config(mem_config)
        self.mem_config = mem_config
        self.cacti_config = CactiConfig()

        if not os.path.isdir(self.CACTI_CONFIG_PATH):
            os.makedirs(self.CACTI_CONFIG_PATH, exist_ok=True)

        if os.path.exists(self.CACTI_OUTPUT_FILE):
            os.system(f'rm {self.CACTI_OUTPUT_FILE}')

    def run_cacti(self):
        #################### 1. CACTI configuration file ####################
        cacti_config_file = os.path.join(self.CACTI_CONFIG_PATH, 'cache.cfg')
        self._prepare_config_file(cacti_config_file)
        
        #################### 2. Run CACTI ####################
        original_cwd = os.getcwd()
        os.chdir(self.CACTI_PROGRAM_PATH) # Change the directory to the cacti program directory
        stream = Popen(f'./cacti -infile {cacti_config_file}', shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = stream.communicate()
        os.chdir(original_cwd) # Change back to the original working directory

        #################### 3. Collect results ####################
        result = {}
        with open(self.CACTI_OUTPUT_FILE, 'r') as fp:
            attribute_list = fp.readline().split(',')
            attribute_list = [s.strip() for s in attribute_list]
            results = dict.fromkeys(attribute_list)

            value_list = fp.readline().split(',')
            value_list = [s.strip() for s in value_list]
            for j, value in enumerate(value_list):
                try:
                    result[attribute_list[j]] = value
                except:
                    pass
        
        bank_count = int(self.mem_config['bank_count'])
        size_byte = result['Capacity (bytes)']
        area = result['Area (mm2)']
        latency = result['Access time (ns)']
        read_energy = result['Dynamic read energy (nJ)']
        write_energy = result['Dynamic write energy (nJ)']
        mem_bw = result['Output width (bits)']

        new_result = {
            'technology': self.mem_config['technology'],
            'size': int(size_byte) * 8 * bank_count, # size in number of bits
            'bank_count': bank_count, 
            'mem_type': self.mem_config['mem_type'],
            'r_port': self.mem_config['r_port'],
            'w_port': self.mem_config['w_port'],
            'rw_port': self.mem_config['rw_port'],
            'rw_bw': self.mem_config['rw_bw'],
            'latency': float(latency),
            'area': float(area) * bank_count,
            'r_cost': float(read_energy) * bank_count * 1000,
            'w_cost': float(write_energy) * bank_count * 1000,
        }

        return new_result

    def _check_valid_config(self, mem_config):
        required_keys = ['technology', 'mem_type', 'size', 'bank_count', 'rw_bw', 'r_port', 'w_port', 'rw_port']
        provided_keys = mem_config.keys()
        missed_keys = []
        for key in required_keys:
            if key not in provided_keys:
                missed_keys.append(key)
        if len(missed_keys) != 0:
            raise ValueError(f'missed keys {missed_keys} from the provided mem_config input')
    
    def _prepare_config_file(self, cacti_config_file):
        mem_config  = {}
        mem_config['technology']   = float(self.mem_config['technology'])
        mem_type = self.mem_config['mem_type']
        mem_config['mem_type']     = f'\"{mem_type}\"'
        mem_config['bank_count']   = 1
        mem_config['cache_size']   = int(self.mem_config['size'] / 8 / self.mem_config['bank_count'])
        mem_config['IO_bus_width'] = int(self.mem_config['rw_bw'] / self.mem_config['bank_count']) 
        mem_config['ex_rd_port']   = self.mem_config['r_port']
        mem_config['ex_wr_port']   = self.mem_config['w_port']
        mem_config['rd_wr_port']   = self.mem_config['rw_port']        

        user_config = []
        cacti_config_option = self.cacti_config.config_option
        if mem_config['IO_bus_width'] < (cacti_config_option['line_size']['default'] * 8) / 2:
            self.cacti_config.change_default_value({'line_size': math.ceil(mem_config['IO_bus_width'] / 64) * 8 })

        for itm in cacti_config_option.keys():
            if itm in mem_config.keys():
                user_config.append(cacti_config_option[itm]['string'] + str(mem_config[itm]) + '\n')
            else:
                user_config.append(cacti_config_option[itm]['string'] + str(cacti_config_option[itm]['default']) + '\n')

        with open(cacti_config_file, 'w+') as f:
            f.write(''.join(self.cacti_config.baseline_config))
            f.write(''.join(user_config))