include("../Code_for_exact_inversion.jl")
using Plots;

parameters = get_parameters()
phys_indices = get_physical_indices()

# set MPS init param
a_array = [i/10 for i in 1:10]
a = a_array[parse(Int64,ARGS[1])+1]
parameters["small initialization parameter"] = a
parameters["number of data points"] = 300

num_of_data_sets = 100
bond_dim_arr_len = 30
bond_dim_arr = [i for i in 1:bond_dim_arr_len]
all_test_losses = Array{Float64}(undef, num_of_data_sets, bond_dim_arr_len) 
all_train_losses = Array{Float64}(undef, num_of_data_sets, bond_dim_arr_len) 
get_data_no_noise(num_of_data_sets, [parameters["number of data points"]])

updates_file = open("txt_updates/a_$(a)", "w")
println(updates_file, "Started inversion for a = $(a) \n")
flush(updates_file)

for i in 1:num_of_data_sets
    
    println("i = $i")
    parameters["sample"] = i
    W = exact_inv()
    test_losses = []
	train_losses = []

    for d in bond_dim_arr
        M = MPS(W, phys_indices; cutoff=0, maxdim=d)
        test_loss_sample = test_loss(M, phys_indices)
		train_loss_sample = train_loss(M, phys_indices)
        append!(test_losses, test_loss_sample)
		append!(train_losses, train_loss_sample)
    end

    all_test_losses[i,:] = test_losses
	all_train_losses[i,:] = train_losses
	println(updates_file, "Finished $(i) data set")
    flush(updates_file)
end


m_test = zeros(bond_dim_arr_len)
msq_test = zeros(bond_dim_arr_len)
std_test = zeros(bond_dim_arr_len)
m_train = zeros(bond_dim_arr_len)
msq_train = zeros(bond_dim_arr_len)
std_train = zeros(bond_dim_arr_len)

for i in 1:bond_dim_arr_len
    m_test[i] = sum(all_test_losses[:,i]) / num_of_data_sets
	msq_test[i] = sum(all_test_losses[:,i].^2) / num_of_data_sets

	m_train[i] = sum(all_train_losses[:,i]) / num_of_data_sets
	msq_train[i] = sum(all_train_losses[:,i].^2) / num_of_data_sets
end

for i in 1:bond_dim_arr_len
    std_test[i] = sqrt((msq_test[i] - m_test[i]^2) / num_of_data_sets)
	std_train[i] = sqrt((msq_train[i] - m_train[i]^2) / num_of_data_sets)
end


# saving results
a = parameters["small initialization parameter"]
Ntr = parameters["number of data points"]
f = parameters["feature mapping dimensionality"]
D = parameters["original feature space dim"]

d = Dict()
d["test mean"] = m_test
d["test std"] = std_test
d["train mean"] = m_train
d["train std"] = std_train
name = "pseudoinv_losses_Ntr_$(Ntr)_a_$(a)"
path = "./Results/$(name)"

jldopen(path, "w") do file
    write(file, name, d) 
end

results = jldopen(path, "r") do file
    read(file, name)
end
