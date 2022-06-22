using JLD
include("Code_for_exact_inversion.jl");

parameters = get_parameters()
phys_indices = get_physical_indices()

bond_dim = 30
parameters["small initialization parameter"] = 0.3
generate_MPS(phys_indices, bond_dim)
get_init_MPS(phys_indices, bond_dim)

num_samples = 100
Ntr_arr = [50 + 50*i for i in 0:16]
get_data_no_noise(num_samples, Ntr_arr);
