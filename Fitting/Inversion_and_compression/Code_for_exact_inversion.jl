using ITensors
using Random, RandomMatrices
import Distributions: MvNormal, Normal, Uniform
# using Dates
using Folds
using LinearAlgebra
using Statistics
using OptimKit # https://github.com/Jutho/OptimKit.jl
using HDF5, JLD
using ITensors.HDF5
BLAS.set_num_threads(4)
ITensors.Strided.disable_threads()
Random.seed!(3)

function get_parameters()
    parameters = Dict( "original feature space dim" => 6, 
                  "feature mapping dimensionality" => 3, 
                  "number of data points" => 128, # training points
                  "number of test points" => 1024, # test points 
                  "L2 regularization param" => 1e-6, 
                  "max bond dimension" => 30, 
                  "small initialization parameter" => 0.3, # used to reduce the effect of higher order terms
                  "MPS initial spoiling" => 0.0,
                  "mps noise weigth" => 0.0,
				  "pseudoinverse cutoff" => 0.0,
				  "sample" => 1
    )
    return parameters
end;


function get_physical_indices(parameters=parameters)
    """
    Generates physical indices for W and Phi tensors, 
    i.e. indices that are common for W and Phi tensors, and 
    which dimension is fixed
    """
    D = parameters["original feature space dim"]
    index_dim = parameters["feature mapping dimensionality"]
    indices = Array{Index}(undef, D)
    for i in 1:D
        indices[i] = Index(index_dim)
    end
    return indices
end;



function get_data_no_noise(num_samples, Ntr_arr, parameters=parameters, phys_indices=phys_indices)
    
    # mkdir data
    if ! isdir("./test_train_data")
        mkpath("./test_train_data")
    end
	path = "./test_train_data/"
    
    D = parameters["original feature space dim"] 
    a = parameters["small initialization parameter"]

	tot_num_of_points = num_samples * Int(sum(Ntr_arr))
	tot_num_of_points += 2100
	Random.seed!(1)

	# Seting up multivariate Gaussian distribution for data
	covar_matr = I  # get_covar_matrix(D, l)
	mean_vect = zeros(D) 
	rand_norm = MvNormal(mean_vect, covar_matr)

	# generating data points
	data = []
	labels = []
	println("Generating data")
	tot_num_of_points = 765000 # remove me?
	for i in 1:tot_num_of_points
	    x = rand(rand_norm)
	    y = get_labels_from_MPS(x, "test") # no noise in test 
	    to_append = [[x, y]] 
	    append!(data, to_append)
	    append!(labels, y)
		# print("generated $(i) point out of $(tot_num_of_points) \r")
	end
	println("Finished generating data")

	# standardize labels
	mean_label = mean(labels)
	std_label = std(labels)
	for (i, (vect, label)) in enumerate(data)
	    data[i][2] = (label - mean_label) / std_label
	end
	println("Labels standardized")

	# Saving test & validation data sets
	saveData(data[1:1024], path, "test_data_a_$(a)_D_$(D)")
	saveData(data[1025:2048], path, "validation_data_a_$(a)_D_$(D)")
	println("test and valid data sets saved")

	# Saving training data sets
	left = 2049
	for i in 1:num_samples
		for Ntr in Ntr_arr
			name = "train_data_a_$(a)_D_$(D)_sample_$(i)"
			right = left + Ntr
			saveData(data[left : right - 1], path, name)
			left = right
		end
		println("saved $(i) data sample out of 100")
	end
end
    
    
function saveData(data, path, name)
    N = length(data)
    name *= "_$(N)_points"
    path *= name * ".jld"
    jldopen(path, "w") do file
        write(file, name, data) 
    end
end


function create_Phi_vectors_for_all_data_points(data, phys_indices=phys_indices, parameters=parameters)
    
    """
    This function returns a vector of phi-vectors (tensors) for all data points, 
    i.e. it returns Phi[i][j], 
    where i corresponds to i-th data point in a data set, 
    j corresponds to j-th component of a data point in a D-dimensional space. 
    
    So, Phi[i][j] is a phi-vector (ITensor object) with a single index of 
    dimension n (feature map dimension), which corresponds to j-th component 
    of an i-th data point   
    """

    n = parameters["feature mapping dimensionality"]
    D = parameters["original feature space dim"]
    
    Phi_all_points = []
    labels = []
    for data_point in data
        x, label = data_point
        single_phi = []
        for (j, point_component) in enumerate(x)
            index = phys_indices[j]
            a = ITensor(index)
            x_j = x[j]
            for k in 1:n
                a[index => k] = x_j^(k - 1)
            end
            single_phi = vcat(single_phi, a)
        end
        Phi_all_points = vcat(Phi_all_points, [single_phi])
        append!(labels, label)
    end
    
    return Phi_all_points, labels
end;



function MPS_grad_and_norm_sq(W)
    
    """
    Calculates full MPS norm |W^2| and a corresponding gradient
    
    Input:
    1. An array of MPS tensors
    
    Output:
    1. gradient of |W|^2
    2. MPS norm squared |W|^2
    """
    
    N = length(W)
    
    # left contructions
    left_contractions = Array{ITensor}(undef, N)
    left = W[1] * prime(W[1], commonind(W[1], W[2]))
    left_contractions[1] = left
    for i in 2:(N-1)
        left *= W[i]
        left *= prime(W[i], commonind(W[i], W[i-1]), commonind(W[i], W[i+1]))
        left_contractions[i] = left
    end
    
    # right contructions
    right_contractions = Array{ITensor}(undef, N)
    right = W[N] * prime(W[N], commonind(W[N], W[N-1]))
    right_contractions[N] = right
    for i in reverse(2:N-1)
        right *= W[i]
        right *= prime(W[i], commonind(W[i], W[i-1]), commonind(W[i], W[i+1]))
        right_contractions[i] = right
    end
    
    
    # setting up a gradient
    grad = W * 0 # Array{ITensor}(undef, N)
    
    # first deriv
    gr = W[1] * right_contractions[2]
    grad[1] = noprime(gr)
    
    # next N-2 deriv
    for i in 2:(N-1)
        gr  = left_contractions[i-1]
        gr *= W[i]
        gr *= right_contractions[i+1]
        grad[i] = noprime(gr)
    end 
    
    # last deriv
    gr = left_contractions[N-1] * W[N]
    grad[N] = noprime(gr)
        
    
    # setting up |W|^2, no need in a separate function
    norm_sq = left_contractions[1] * right_contractions[2]
    

    return 2 * grad, norm_sq[]
    
end;
    

function loss(A_tensors, phi_vectors, labels, phys_indices)
    
    # get loss function
    loss = Folds.reduce(+, (loss_per_sample(A_tensors, phi, label, phys_indices) for (label, phi) in 
                                    zip(labels,phi_vectors)))
    loss /= (2*length(phi_vectors))   
    return loss 
end


function loss_per_sample(A_tensors, phi_D, label, phys_indices)
	# sum over the number of points
	loss = 0.0
	f = 1
    # product of all phi and W tensors
    for (j, phi) in enumerate(phi_D)
        f = f * A_tensors[j] * phi
    end
    loss += (f[] - label)^2
	return loss
end


function test_loss(W_tensors, phys_indices, parameters=parameters)
    
    """
    Returns test loss
    """

    D = length(W_tensors)
    N = parameters["number of test points"]
    
    # get test data and test loss
    D = parameters["original feature space dim"] 
    a = parameters["small initialization parameter"]
	# alpha = parameters["mps noise weigth"]
	L2 = parameters["L2 regularization param"]
	sample = parameters["sample"]
    name = "test_data_a_$(a)_D_$(D)"
	name *= "_$(N)_points"
    path = "./test_train_data/"*name*".jld"
    test_data = jldopen(path, "r") do file
        read(file, name)
    end

	# save old parameters
    lam = parameters["L2 regularization param"]
    parameters["L2 regularization param"] = 0.0		

    phi_test, labels_test = create_Phi_vectors_for_all_data_points(test_data)
    test_loss_val = loss(W_tensors, phi_test, labels_test, phys_indices)
    
    # return to finite L_2
    parameters["L2 regularization param"] = lam
    
    return test_loss_val
end


function train_loss(W_tensors, phys_indices, parameters=parameters)
    
    """
    Combine test and train losses later
    """

    D = length(W_tensors)
    N = parameters["number of data points"]
    
    # get test data and test loss
    D = parameters["original feature space dim"] 
    a = parameters["small initialization parameter"]
    # alpha = parameters["mps noise weigth"]
	L2 = parameters["L2 regularization param"]
	sample = parameters["sample"]
    name = "train_data_a_$(a)_D_$(D)_sample_$(sample)"
    name *= "_$(N)_points"
    path = "./test_train_data/"*name*".jld"
    test_data = jldopen(path, "r") do file
        read(file, name)
    end

    phi_test, labels_test = create_Phi_vectors_for_all_data_points(test_data)
    train_loss_val = loss(W_tensors, phi_test, labels_test, phys_indices)
    
    return train_loss_val
end


function get_W_matrices_for_lor_ord_pol_init(n)
    
    """
    Returns nilpotent matrices for setting up an MPS returning an n-th 
    order polynomial
    """ 
    
    dim = n + 1
    
#     # get leftmost tensor
#     L = UpperTriangular(ones(dim, dim))
    
    # get middle tensors (nilpotent matrices)
    A = Matrix(1.0I, dim, dim)
    mid = zeros(dim, dim, dim)
    mid[1,:,:] = A
    B_0 = diagm(1 => [1.0 for i in 1:dim-1])
    U = rand(Haar(1), dim) # unitary matrix
    B_0 = U * B_0 * transpose(U) # more general nilpotent matrix
    B = B_0
    for i in 2:dim
        mid[i,:,:] = B
        B = B * B_0
    end

#     # get rightmost tensor
#     R = rotl90(L)
    
    path = "../MPS/"
    if ! isdir(path)
        mkpath(path)
    end
    
	name = "nilpot"
    jldopen(path * name * ".jld", "w") do file
        write(file, name, mid) 
    end

	return mid
end;


function generate_MPS(phys_indices, bond_dim, parameters=parameters)
    
    """
    Run this once in the beginning before any other calculations 
    to generate MPS for training and testing
    """
    
    D = parameters["original feature space dim"]
    d = bond_dim
    f = parameters["feature mapping dimensionality"]
    parameters["max bond dimension"] = bond_dim
    
    A_matrices = Array{ITensor}(undef, D) # matrices entering W-tensor MPS
    W_random = Array{ITensor}(undef, D)
    path_nilpot = "../MPS/nilpot.jld"
    if ! isfile(path_nilpot)
        z = get_W_matrices_for_lor_ord_pol_init(d-1)
    else
        z = jldopen(path_nilpot, "r") do file
            read(file, "nilpot")
        end
    end

    # leftmost tensor
    i_right = Index(d)
    W_random[1] = randomITensor(phys_indices[1], i_right)
    
    # random (noise) tensors in the bulk
    for s in 2:(D-1)
		i_left = i_right
        i_right = Index(d)
        W_random[s] = randomITensor(i_left, phys_indices[s], i_right)
    end
	W_random[end] = randomITensor(i_right, phys_indices[end])
    
    # save noisy MPS
    save_MPS(W_random, "../MPS/noisy_MPS_D_$(D)_bond_$(d)_f_$(f)")


    # middle "clean" tensors for each value of a
	rand_matr_left = rand(Normal(0,1), f, d)
	rand_matr_right = rand(Normal(0,1), d, f)
    for a in [i/10 for i in 1:10] # iterate over each value of a
		i_right = Index(d)
    	A_matrices[1] = ITensor(rand_matr_left, phys_indices[1], i_right) 
        for s in 2:(D-1) # enumerting tensors in the bulk
            i_left = i_right
            i_right = Index(d)
            A = ITensor(i_left, phys_indices[s], i_right)
            for i in 1:d
                for j in 1:3 # as we want only 1, x, x^2, but not x^3, x^4 and so on
                    for k in 1:d
                        A[i,j,k] = z[j,i,k] * a^(j-1)
                    end
                end
            end
            A_matrices[s] = A 
        end
		A_matrices[end] = ITensor(rand_matr_right, i_right, phys_indices[end])
        # save corresponding clean MPS
        save_MPS(A_matrices, "../MPS/MPS_a_$(a)_D_$(D)_bond_$(d)_f_$(f)")
    end
end


function get_init_MPS(phys_indices, bond_dim, parameters=parameters)

	if ! isdir("./test_train_data")
		mkpath("./test_train_data")
	end

    D = parameters["original feature space dim"]
    d = bond_dim
    f = parameters["feature mapping dimensionality"]
    parameters["max bond dimension"] = bond_dim
    a = parameters["small initialization parameter"]
    alpha = parameters["mps noise weigth"]
    
    clean_MPS = load_MPS("../MPS/MPS_a_$(a)_D_$(D)_bond_$(d)_f_$(f).h5")
    noise = load_MPS("../MPS/noisy_MPS_D_$(D)_bond_$(d)_f_$(f).h5")

	if ! isfile("./test_train_data/clean_MPS_a_$(a).h5")
		# save clean MPS
		save_MPS(clean_MPS, "./test_train_data/clean_MPS_a_$(a)")
	end
    
	if ! isfile("./test_train_data/noisy_MPS_a_$(a)_alpha_$(alpha).h5")
		# save noisy MPS
		noise_correct_inds = deepcopy(clean_MPS)
		for i in 1:f
		    for j in 1:d
		        noise_correct_inds[1][i,j] = noise[1][i,j]
		    end
		end
		for s in 2:(D-1)
		    for i in 1:d
		        for j in 1:f
		            for k in 1:d
		                noise_correct_inds[s][i,j,k] = noise[s][i,j,k]
		            end
		        end
		    end
		end
		for i in 1:d
		    for j in 1:f
		        noise_correct_inds[D][i,j] = noise[D][i,j]
		    end
		end
		noisy_mps = clean_MPS + alpha * noise_correct_inds
		save_MPS(noisy_mps, "./test_train_data/noisy_MPS_a_$(a)_alpha_$(alpha)")
	end
end
    
    
    

function get_labels_from_MPS(data, test_or_train, parameters=parameters, phys_indices=phys_indices)

    # load label-generating MPS 
    a = parameters["small initialization parameter"]
    alpha = parameters["mps noise weigth"]
    if test_or_train == "train"
        given_MPS = load_MPS("./test_train_data/noisy_MPS_a_$(a)_alpha_$(alpha).h5")
    elseif test_or_train == "test"
        given_MPS = load_MPS("./test_train_data/clean_MPS_a_$(a).h5")
    end
    
    f = parameters["feature mapping dimensionality"]
    D = parameters["original feature space dim"]
    W = 1
    
    for (i,x) in enumerate(data)
        # set up data tensor
        p = ITensor(phys_indices[i])
        for j in 1:f
            p[j] = x^(j-1)
        end
        # contract MPS and data tensors
        W *= p
        W *= given_MPS[i]
    end
    return W[]
end;


function save_MPS(W_tensors, file_path_name,
                  phys_indices=phys_indices, 
                  parameters=parameters)

    # saving an MPS
    D = parameters["original feature space dim"]
    d = parameters["max bond dimension"]
    f = parameters["feature mapping dimensionality"]

    # 1st index - number/site/position of a tensor
    # 2nd index - virtual index
    # 3d index - physical index
    # 4th index - virtual index
    W_tensors_to_save = Array{Float64}(undef, D, d, f, d);

    v1 = commonind(W_tensors[1], W_tensors[2])
    W = Array(W_tensors[1], phys_indices[1], v1)
    W_tensors_to_save[1, 1,:,:] = W

    for i in 2:(D-1)
        v_left = commonind(W_tensors[i], W_tensors[i-1])
        v_right = commonind(W_tensors[i], W_tensors[i+1])
        W = Array(W_tensors[i], v_left, phys_indices[i], v_right)
        W_tensors_to_save[i,:,:,:] = W
    end

    W = Array(W_tensors[end], commonind(W_tensors[D-1], W_tensors[D]), phys_indices[end])
    W_tensors_to_save[end, :,:,1] = W

    f = h5open(file_path_name*".h5", "w")
    write(f, "saved_MPS", W_tensors_to_save)
    close(f)
    
end;


function load_MPS(file_path_name,
                  phys_indices=phys_indices, 
                  parameters=parameters)
    
    # loading an MPS
    f = h5open(file_path_name, "r")
    T = read(f, "saved_MPS")
    close(f)

    D = parameters["original feature space dim"]
    W_loaded = Array{ITensor,1}(undef, D)
    # W_loaded[1] = ITensor(T[1,1,:,:], phys_indices[1], virt_indices[1])
    v_ind = Index(length(T[1,1,1,:]))
    W_loaded[1] = ITensor(T[1,1,:,:], phys_indices[1], v_ind)
    v_left = v_ind
    v_right = 1

    for i in 2:D-1
        # v_left = Index(length(T[i,:,1,1]))
        v_right = Index(length(T[i,1,1,:]))
        # W = ITensor(T[i,:,:,:], virt_indices[i-1], phys_indices[i], virt_indices[i])
        W = ITensor(T[i,:,:,:], v_left, phys_indices[i], v_right)
        W_loaded[i] = W
        v_left = v_right
    end

    W_loaded[end] = ITensor(T[end, :,:,1], v_right, phys_indices[end])
    
    return W_loaded
end;


function MPS_norm_squared(W)
    
    """
    Calculates MPS norm
    """
    
    N = length(W)

    # first step 
    result = W[1]
    result *= prime(W[1], commonind(W[1], W[2]))
    
    # next N-2 steps
    for i in 2:(N-1)
        result *= W[i]
        result *= prime(W[i], commonind(W[i], W[i-1]), commonind(W[i], W[i+1]))
    end
    
    # last step 
    result *= W[N]
    result *= prime(W[N], commonind(W[N], W[N-1]))
    
    return result[]
end;


function right_canonical_form(W, phys_indices)
    
    """
    Returns a right-canonical MPS
    """
    
    N = length(W)

    # rightmost tensor
    # U, S, V = svd(W[N], (phys_indices[N]); maxdim=bond_dim, lefttags = "", righttags = "");
    Q, R = qr(W[N], (phys_indices[N]))
    W[N] = Q
    W[N-1] = R * W[N-1]

    # middle tensors, going from the right to the left
    for i in reverse(2:N-1)
        virt_ind = commonind(W[i], W[i+1])
        # U, S, V = svd(W[i], (virt_ind, phys_indices[i]); maxdim=bond_dim, lefttags = "", righttags = "")
        Q, R = qr(W[i], (virt_ind, phys_indices[i]))
        W[i] = Q
        W[i-1] = R * W[i-1]
    end

    return W
end; 


function MPS_compression(W_tens, phys_indices, new_d)
    
    """
    Given a general MPS, returns a compressed one in a left-canonical form 
    with smaller bond dimension new_d
    """
    
    # prepare an MPS in a right-canonical form first
    W_rc = right_canonical_form(W_tens, phys_indices);
    
    # copy MPS to compress it
    W = [copy(W_rc[i]) for i in 1:length(W_rc)];
    N = length(W)

    # middle tensors, going from the right to the left
    # sweep from left to right
    for i in 1:N-1
        B = W[i] * W[i+1]
        if i == 1
            U, S, V = svd(B, (phys_indices[i]); 
                            maxdim = new_d, lefttags = "", righttags = "")
        else
            virt_ind = commonind(W[i], W[i-1])
            U, S, V = svd(B, (virt_ind, phys_indices[i]); 
                            maxdim = new_d, lefttags = "", righttags = "")
        end 
        W[i] = U
        W[i+1] = S * V
    end

    return W
end;   


function pseudoinverse(M, b_vector, cut_off)
    # see built-in pinv function in https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/
    svd_M = svd(M)
    # M = svd_A.U * Diagonal(svd_A.S) * svd_A.Vt
    s_n = length(svd_M.S)
    S_inv = zeros(s_n, s_n)
    
    for i in 1:s_n
        if svd_M.S[i] >= cut_off
            S_inv[i,i] = 1.0 / svd_M.S[i]
        end
    end
    pseudo_inv = transpose(svd_M.Vt) * S_inv * transpose(svd_M.U) * b_vector
    return pseudo_inv
end;


function get_diag_inds(all_n, set, prefix, n, k)
     
    # Base case: k is 0
    if k == 0
        append!(all_n, [vcat(prefix, prefix)])
        return
    end
 
    # One by one add all characters
    # from set and recursively
    # call for k equals to k-1
    for i in 1:n
 
        # Next character of input added
        newPrefix = vcat(prefix, set[i])
         
        # k is decreased, because
        # we have added a new character
        get_diag_inds(all_n, set, newPrefix, n, k - 1)
    end
end


function exact_inv(approach="backslash", phys_indices=phys_indices, parameters=parameters)
    
    N = parameters["number of data points"]
    D = parameters["original feature space dim"]
    f = parameters["feature mapping dimensionality"]
    a = parameters["small initialization parameter"]
	alpha = parameters["mps noise weigth"]
	L2 = parameters["L2 regularization param"]
    
    # get data
    sample = parameters["sample"]
    name = "train_data_a_$(a)_D_$(D)_sample_$(sample)"
    name *= "_$(N)_points"
    path = "./test_train_data/"*name*".jld"
    data = jldopen(path, "r") do file
        read(file, name)
    end
    
    # set up phys indices, phi data vectors and labels
    phi_all, labels = create_Phi_vectors_for_all_data_points(data, phys_indices)

    # setting up exact inversion
    a1 = [phys_indices[i] for i in 1:length(phys_indices)]
    a2 = [prime(phys_indices[i]) for i in 1:length(phys_indices)]
    A_indices = vcat(a1, a2) 
    A_tens = ITensor(A_indices...)
    b_tens = ITensor(phys_indices...)
    
    # iterating over samples
    for (phi, y) in zip(phi_all, labels)
        # iterating over physical (index) positions
        a, b = 1, y
        for i in 1:D
            a *= phi[i] 
            b *= phi[i]
        end
        for i in 1:D 
            a *= prime(phi[i]) 
        end
        A_tens += a
        b_tens += b
    end
    
    # normalize 
    A_tens /= length(labels)
    b_tens /= length(labels)
    
    # turn A and b into arrays
    A_arr = Array(A_tens, A_indices...)
    b_arr = Array(b_tens, phys_indices...)
    
    # add regularization to the A matrix, 
    # i.e. to its "diagonal" part A[i,j,k,l,..., i,j,k,l,...]
    lam = A_arr * 0
    l = parameters["L2 regularization param"]
    all_n = []
    set = [i for i in 1:f]
    k = length(phys_indices)
    get_diag_inds(all_n, set, [], length(set), k)
    for i in all_n
        lam[i...] = l 
    end
    A_arr += lam

    # reshape them, so A is a matrix and b is a vector
    n = f^D
    A_matrix = reshape(A_arr, (n, n)) 
    b_vector = reshape(b_arr, (n)) 
    

    # get full W tensor (as a matrix)
    cut_off = parameters["pseudoinverse cutoff"]
    W_matr = pseudoinverse(A_matrix, b_vector, cut_off)
	if approach == "SVD"
#         W_matr = pseudoinverse(A_matrix, b_vector, cut_off)
	elseif approach == "backslash"
		W_matr = A_matrix \ b_vector
	elseif approach == "optim"
		w_init = A_matrix \ b_vector
#         w_init = rand(729)
 		lg(w) = 0.5 * sum((A_matrix * w - b_vector).^2), (A_matrix * w - b_vector)
		W_matr, fx, gx, numfg, normgradhistory = optimize(lg, w_init, 
                                                     ConjugateGradient(;maxiter = 1000, verbosity = 2))
	else
		throw(DomainError(approach, "must a string be either SVD, or backslash, or optim"))
	end

    # reshape W_matr into a W-tensor
    W_tensor = ITensor(W_matr, phys_indices...)
    
    return W_tensor #MPS(W_tensor, phys_indices; cutoff=0, maxdim=100) 
    
end


  
    
