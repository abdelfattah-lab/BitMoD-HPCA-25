import argparse
from accelerator import Accelerator 

model_list = ["facebook/opt-1.3b", "microsoft/phi-2", "01-ai/Yi-6B", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_generation", action="store_true", help="If enabled, then evaluate")
    parser.add_argument("--is_lossless", action="store_true", help="If enabled, then evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="The input batch size")
    parser.add_argument("--cxt_len", type=int, default=256, help="The input context length")

    args = parser.parse_args()
    is_generation = args.is_generation
    is_lossless   = args.is_lossless
    batch_size    = args.batch_size
    cxt_len       = args.cxt_len

    assert batch_size > 0, "The input batch_size must be > 1"

    #################### Set Precision and PE array characteristic ####################
    pe_dp_size = 4
    is_bit_serial = True
    pe_energy = 0.56
    pe_area   = 1507.7
    if is_generation:
        pe_array_dim = [64, 16]
    else:
        pe_array_dim = [32, 32]
    
    #################### Set Precision ####################
    kv_prec = {}
    for model_name in model_list:
        kv_prec[model_name] = 8

    if is_generation:
        if is_lossless:
            w_prec = 6.0625
        else:
            w_prec = 3.0625
    else:
        if is_lossless:
            w_prec = 6.0625
        else:
            w_prec = 4.0625
    
    #################### Simulate Perforamnce and Energy ####################
    total_energy_list = [[0, 0] for _ in model_list]
    total_latency_list = [0 for _ in model_list]

    for idx, model_name in enumerate(model_list):
        acc = Accelerator(
            model_name=model_name, 
            i_prec=16,
            kv_prec=kv_prec[model_name],
            w_prec=w_prec,
            batch_size=batch_size,
            is_bit_serial=is_bit_serial,
            pe_dp_size=pe_dp_size,
            pe_energy=pe_energy,
            pe_area=pe_area,
            pe_array_dim=pe_array_dim,
            cxt_len=cxt_len,
            is_generation=is_generation,
        )

        total_cycle    = acc.calc_cycle()
        compute_energy = acc.calc_compute_energy() / 1e6
        sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
        sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
        dram_energy    = acc.calc_dram_energy() / 1e6
        onchip_energy  = compute_energy + sram_rd_energy + sram_wr_energy
        total_energy   = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy

        print(f'model: {model_name}')
        print(f'total cycle:        {total_cycle}')
        total_latency_list[idx] = total_cycle[1]

        print(f'pe array area:      {acc.pe_array_area / 1e6} mm2')
        print(f'weight buffer area: {acc.w_sram.area} mm2')
        print(f'input buffer area:  {acc.i_sram.area} mm2')
        # print(f'compute energy:     {compute_energy} uJ')
        # print(f'sram rd energy:     {sram_rd_energy} uJ')
        # print(f'sram wr energy:     {sram_wr_energy} uJ')
        print(f'dram energy:        {dram_energy} uJ')
        print(f'on-chip energy:     {onchip_energy} uJ')
        print(f'total energy:       {total_energy} uJ')
        total_energy_list[idx][0] = round(onchip_energy)
        total_energy_list[idx][1] = round(total_energy)
        
        print('\n')
    
    print(f'Latency: {total_latency_list}')
    print(f'Energy: {total_energy_list}')
    