include("../Code_for_exact_inversion.jl")

parameters = get_parameters()
phys_indices = get_physical_indices()

# set MPS init param
Ntr_arr = [50 + 50*i for i in 0:16]
bond_dim_arr = [i for i in 2:30]
bond_dim_arr_len = length(bond_dim_arr)
num_of_data_sets = 100
Ntr = Ntr_arr[parse(Int64,ARGS[1])+1]
parameters["number of data points"] = Ntr
updates_file = open("txt_updates/Ntr_$(Ntr)", "w")
println(updates_file, "Started inversion for $(Ntr) training points \n")
flush(updates_file)

all_test_losses = Array{Float64}(undef, num_of_data_sets, bond_dim_arr_len) 
all_train_losses = Array{Float64}(undef, num_of_data_sets, bond_dim_arr_len) 


for i in 1:num_of_data_sets
    # iterate over different data sets
	parameters["sample"] = i
    W = exact_inv()
    test_losses = []
	train_losses = []

    for d in bond_dim_arr
		# iterate over bond dimensions
		parameters["max bond dimension"] = d
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

println(updates_file, "\n\n")
flush(updates_file)


m_test    = zeros(bond_dim_arr_len)
msq_test  = zeros(bond_dim_arr_len)
std_test  = zeros(bond_dim_arr_len)
m_train   = zeros(bond_dim_arr_len)
msq_train = zeros(bond_dim_arr_len)
std_train = zeros(bond_dim_arr_len)
for i in 1:bond_dim_arr_len
	m_test[i] = sum(all_test_losses[:,i]) / num_of_data_sets
	m_train[i] = sum(all_train_losses[:,i]) / num_of_data_sets
end


if Ntr < 3^6
	for i in 1:bond_dim_arr_len
		msq_test[i] = sum(all_test_losses[:,i].^2) / num_of_data_sets
		msq_train[i] = sum(all_train_losses[:,i].^2) / num_of_data_sets
	end

	for i in 1:bond_dim_arr_len
		std_test[i] = sqrt((msq_test[i] - m_test[i]^2) / num_of_data_sets)
		std_train[i] = sqrt((msq_train[i] - m_train[i]^2) / num_of_data_sets)
	end
end

# saving results
a = parameters["small initialization parameter"]
Ntr = parameters["number of data points"]

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

println(updates_file, "Done!")
flush(updates_file)

#results = jldopen(path, "r") do file
#    read(file, name)
#end
