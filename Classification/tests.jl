include("OptimKit_correction.jl")
include("MPS_MPO_scripts.jl")
include("GetingData.jl")
include("Optimization.jl")

parameters = getParameters()
parameters["training_set_size"] = 1024
parameters["num_of_epochs"] = 2000
parameters["MPO_bond_dim"] = 1

noise_arr = [0.0, 0.1, 0.2]
chi_arr = [i for i in 2:20]

parameters["label_noise"] = noise_arr[parse(Int64,ARGS[1])+1]
parameters["mps_bond_dim"] = chi_arr[parse(Int64,ARGS[2])+1]


t = @elapsed optimizeFullTN()

updates_file = open(filename(), "a")
println(updates_file, "Total calculation time is $(t)")
flush(updates_file)
close(updates_file)
