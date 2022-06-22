include("../Code_for_exact_inversion.jl")
include("../Code_for_optimization.jl")

# if !isdir("results") mkdir("results") end
parameters = get_parameters()
parameters["small initialization parameter"] = 0.3
parameters["number of data points"] = 300 
seed_arr = [i for i in 1:100]
bond_dim_array = [i for i in 2:30]

parameters["sample"] = seed_arr[parse(Int64,ARGS[1])+1]
parameters["max bond dimension"] = bond_dim_array[parse(Int64,ARGS[2])+1]
chi = parameters["max bond dimension"]

Ntr = parameters["number of data points"]
a = parameters["small initialization parameter"]
sample = parameters["sample"]
name = "Full_optim_losses_Ntr_$(Ntr)_a_$(a)_chi_$(chi)_init_backslash_sample_$(sample)"
if isfile("results/"* name *".jld")
    throw(ErrorException("$(name) already done!")) 
end

phys_indices = get_physical_indices()
phi, labels = create_Phi_vectors_for_all_data_points(get_training_data(), phys_indices)

# setting up initial conditions
W_tensors = get_init_compressed(chi, "backslash") 

# run optimization
num_of_steps = 1e+9
@time x_test, valid_losses, x_train, train_losses = local_update(W_tensors, phi, labels, num_of_steps, "QR");

# save results
jldopen("results/"* name *".jld", "w") do file
	write(file, name, [x_test, valid_losses, x_train, train_losses])  
end
