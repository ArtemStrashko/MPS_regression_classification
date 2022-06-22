include("../Code_for_exact_inversion.jl")

parameters = get_parameters()
phys_indices = get_physical_indices()

# set MPS init param
a_array = [i/10 for i in 1:10]
for a in a_array
	parameters["small initialization parameter"] = a
	generate_MPS(phys_indices, 30)
	get_init_MPS(phys_indices, 30)
end

