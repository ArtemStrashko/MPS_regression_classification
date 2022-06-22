using Folds

function get_rand_or_ident_init_MPS(phys_indices, bond_dim, rand_or_ident, rand_seed=3, parameters=parameters)
    
    """
    Initializes a vector of tensors entering the weight W-tensor MPS. 
    Returns corresponding tensors and also a vector of virtual indices. 

    Leftmost and rightmost tensors are random, middle tensors have 
    only 1s on diagonal and 0s elsewhere for each physical index value. 
    This is a good initialization as it helps to avoid very large or small 
    initial loss function. 
    """
	Random.seed!(rand_seed)
    
    D = parameters["original feature space dim"]
    n = parameters["feature mapping dimensionality"]
    A_matrices = Array{ITensor}(undef, D) # matrices entering W-tensor MPS
    small_num = 1e-1
    
    # leftmost tensor
    i_right = Index(bond_dim)
    A = randomITensor(phys_indices[1], i_right)
    A_matrices[1] = A
    
    # middle tensors
    for i in 2:(D-1)
        i_left = i_right
        i_right = Index(bond_dim)
        # i_left - left virtual ind, indices[i] - phys index, i_right - right virt ind
        if rand_or_ident == "random"
            A = randomITensor(i_left, phys_indices[i], i_right)
        elseif rand_or_ident == "ident"
            # better A initialization
            A = ITensor(i_left, phys_indices[i], i_right)
            for j in 1:n
                for k in 1:bond_dim
                    A[i_left => k, phys_indices[i] => j, i_right => k] = 1 * small_num^j
                end
            end
        end
        A_matrices[i] = A
    end 
    
    # rightmost tensor
    A = randomITensor(i_right, phys_indices[end])
    A_matrices[D] = A
    
    return A_matrices
    
end;  


function get_init_compressed(new_bond_dim, approach, 
                                phys_indices=phys_indices, 
                                parameters=parameters)

	W = exact_inv(approach)
	W = MPS_compression(W, phys_indices, new_bond_dim)
	return W
end
	




function tensor_update(B, l, forward, W_tensors, 
                       method="not canon", 
                       phys_indices=phys_indices, 
                       parameters=parameters)
    
    D = parameters["original feature space dim"]
    chi = parameters["max bond dimension"]
    
    if method == "QR"
        if l == 1
            Q, R = qr(B, (phys_indices[1]))
            W_tensors[l]   = Q 
            W_tensors[l+1] = R * W_tensors[l+1]
        elseif l == D
            Q, R = qr(B, (phys_indices[l]))
            W_tensors[l]   = Q 
            W_tensors[l-1] = R * W_tensors[l-1]
        elseif forward
            Q, R = qr(B, (commonind(W_tensors[l], W_tensors[l-1]), phys_indices[l])) 
            W_tensors[l]   = Q 
            W_tensors[l+1] = R * W_tensors[l+1] 
        elseif !forward
            Q, R = qr(B, (commonind(W_tensors[l], W_tensors[l+1]), phys_indices[l]))
            W_tensors[l]   = Q 
            W_tensors[l-1] = R * W_tensors[l-1]
        end
    elseif method == "SVD"
        if l == 1
            U, S, V = svd(B, (phys_indices[1]), maxdim=chi)
            W_tensors[l]   = U 
            W_tensors[l+1] = W_tensors[l+1] * S * V
        elseif l == D
            U, S, V = svd(B, (phys_indices[l]), maxdim=chi)
            W_tensors[l]   = U 
            W_tensors[l-1] = W_tensors[l-1] * S * V
        elseif forward
            U, S, V = svd(B, (commonind(W_tensors[l], W_tensors[l-1]), phys_indices[l]), maxdim=chi) 
            W_tensors[l]   = U 
            W_tensors[l+1] = W_tensors[l+1]  * S * V
        elseif !forward
            U, S, V = svd(B, (commonind(W_tensors[l], W_tensors[l+1]), phys_indices[l]), maxdim=chi)
            W_tensors[l]   = U 
            W_tensors[l-1] = W_tensors[l-1] * S * V
        end
    elseif method == "not canon"
        W_tensors[l] = B
    end
end


function loss_grad_for_local_update(x, A, b, lam)
    """
    Returns loss and gradient value with respect to a B-tensor 
    (single MPS tensor) for a single site local update

    Input:
    1. B-tensor, i.e. B = W_tensors[i] 
    2. A and b tensors, which meaning is clear from the loss definition below. 
    """
    # loss is defined up to a constant
    loss = (x * prime(x) * A)[] / 2 - (x * b)[] + lam * norm(x)^2
    grad = prime(x) * A - b + 2 * lam * x
    return loss, grad
end;  
    

function local_update(W_tensors, Phi_all_data_points, labels, num_of_steps=1e+5, method="not canon", 
                    num_cgd_steps=10, 
                    phys_indices=phys_indices, 
                    parameters=parameters)
    
    D = parameters["original feature space dim"]
    bond_dim = parameters["max bond dimension"]
    lam = parameters["L2 regularization param"]
	a = parameters["small initialization parameter"]
	sample = parameters["sample"]
	if lam != 0.0
		if ! (method in ["QR", "SVD"])
			 throw(DomainError(method, "must a either QR or SVD when using finite regularization lambda"))
		else
		# bring an MPS to a right-canonical form
        W_tensors = right_canonical_form(W_tensors, phys_indices)
		end
	end

	Ntr = parameters["number of data points"]
    t = 0
    l = 1 # site number
    forward = true # going left (forward or right)
    test_losses = [test_loss(W_tensors, phys_indices)]
    train_losses = [loss(W_tensors, Phi_all_data_points, labels, phys_indices)[1]]
	valid_losses = [valid_loss(W_tensors, phys_indices)]
    
    # auxiliary arrays storing contractions for speeding up calculations
    left_contractions = Array{ITensor}(undef, Ntr, D)
    right_contractions = Array{ITensor}(undef, Ntr, D)

	t0 = 5000
    
    # save train loss every Ntn steps
    Ntn = t0 # 100
    
    # save test loss every Ntt steps
    Ntt = t0 # 100

	last_train = 0.0
	last_test = 0.0
	last_valid = 0.0
	best_valid = 1e+6
	test_corresp_to_best_valid = 1e+6
	
    
    while t < num_of_steps   # 100000 
        
        f = j -> getTensorsForLossGrad_per_sample(j, l, Phi_all_data_points, W_tensors, left_contractions, right_contractions, forward, labels)
        # j in an image index (number of a given image)
        A, b = Folds.reduce(+, (f(j) for j in 1:Ntr))
        # divide A and b tensors by the number of points
        A /= Ntr
        b /= Ntr
        
        # initialize B-tensor
        B_init = W_tensors[l]
        
        # update B-tensor with CGD
        lg = x -> loss_grad_for_local_update(x, A, b, lam)
        B, fx, gx, numfg, normgradhistory = optimize(lg, B_init, 
                                                     ConjugateGradient(;maxiter = num_cgd_steps, verbosity = 0))
        
        # update W tensors keeping MPS in a canonical form
        tensor_update(B, l, forward, W_tensors, method)
        
        # shift site index
        if forward && l < D
            l += 1
        elseif forward && l == D
            l -= 1
            forward = false
        elseif !forward && l > 1
            l -= 1
        elseif !forward && l == 1
            l += 1
            forward = true
        end
        t += 1
        

        # update test, train and valid losses
        if mod(t, Ntn) == 0

            train = loss(W_tensors, Phi_all_data_points, labels, phys_indices)
			test = test_loss(W_tensors, phys_indices)
			valid = valid_loss(W_tensors, phys_indices)
            append!(train_losses, train)
            append!(test_losses, test)
			append!(valid_losses, valid)
      
			# save results
			if valid < best_valid
				best_valid = valid
				test_corresp_to_best_valid = test
			end
			last_train = train
			last_valid = valid
			last_test = test
			
			name = "Point_losses_Ntr_$(Ntr)_a_$(a)_chi_$(bond_dim)_sample_$(sample)"
			jldopen("results/"* name *".jld", "w") do file
				write(file, name, [last_train, last_valid, last_test, best_valid, test_corresp_to_best_valid])  
			end
        end 

		if mod(t, t0) == 0
			println("step $t, train loss $train")
            println("step $t, test loss $test\n")
		end
        
        if ((t > 1e+6 && 
				(abs(train_losses[end] - train_losses[end-100]) / train_losses[end] < 1e-4 || 
				 train_losses[end] > train_losses[end-100])) || t > 1e+7)
            break
        end

    end
    
#     # save resulting mps
#     mps_name = "mps_bond_d_$(bond_dim)_"
#     mps_name *= "Ntr_$(Ntr)"
#     file_path_name = "results/" * mps_name
#     save_MPS(W_tensors, file_path_name)
    
    # for plotting
    x_test = LinRange(0, (length(valid_losses)-1)*Ntt, length(valid_losses))
    x_train = LinRange(0, (length(train_losses)-1)*Ntn, length(train_losses))
    
    return x_test, valid_losses, x_train, train_losses
end


function getTensorsForLossGrad_per_sample(j, l, Phi_all_data_points, W_tensors, left_contractions, right_contractions, forward, labels, parameters=parameters)
	
	D = parameters["original feature space dim"]
    if forward # going to the right
        
        # left part of the Phi_tilda tensor
        # when going forward, ALWAYS build new left_contract
        left_contract = 1
        if l == 1 # nothing to contract
            left_contract = 1
        elseif l == 2 # only one contraction
            left_contract = W_tensors[1] * Phi_all_data_points[j][1]
        else
            left_contract = (left_contractions[j, l - 2] * W_tensors[l - 1] * 
                             Phi_all_data_points[j][l - 1])                       
        end
        
        # middle part of the Phi_tilda tensor
        mid_data = Phi_all_data_points[j][l]

        # right part of the Phi_tilda tensor
        right_contract = 1
        # use saved contraction if exists
        if isassigned(right_contractions, j, l + 1)
            right_contract = right_contractions[j, l + 1]
        else # doesn't exist -> build it
            for r in (l+1):D
                right_contract = (right_contract * 
                                  W_tensors[r] * 
                                  Phi_all_data_points[j][r])
            end
        end
  
    else # going to the left
        
        left_contract = 1
        # use saved contraction if exists
        if isassigned(left_contractions, j, l - 1)
            left_contract = left_contractions[j, l - 1]
        else # doesn't exist -> build it
            for l in 1:l-1
                left_contract = (left_contract * 
                                 W_tensors[l] * 
                                  Phi_all_data_points[j][l])
            end
        end

        # middle part of the Phi_tilda tensor  
        mid_data = Phi_all_data_points[j][l]
                
        # right part of the Phi_tilda tensor
        # when going backward, ALWAYS build new right_contract
        right_contract = 1
        if l == D - 1
            right_contract = W_tensors[D] * Phi_all_data_points[j][D]
        else
            right_contract = right_contract * right_contractions[j, l + 2]
            right_contract = (right_contract * 
                                W_tensors[l + 1] * 
                                Phi_all_data_points[j][l + 1])
        end
    end
    
    # UPDATE CONTRACTIONS for a corresponding data point to use them later
    if (forward && l > 1) 
        left_contractions[j, l - 1] = left_contract
    elseif !forward && l < D
        right_contractions[j, l + 1] = right_contract
    end
    
    # Full Phi-tilda tensor 
    Phi_tilda = left_contract * mid_data * right_contract
    
    # Construct A and b tensors for CGD update of a B-tensor
    # B tensor is similar to one defined in Eq.(7) in 
    # https://arxiv.org/abs/1605.05775 for DMRG calculations
    A = Phi_tilda * prime(Phi_tilda)
    b = labels[j] * Phi_tilda 

	return [A, b]
end


function get_training_data(parameters=parameters)
    Ntr = parameters["number of data points"]
    D = parameters["original feature space dim"]
    a = parameters["small initialization parameter"]
    alpha = parameters["mps noise weigth"]
	L2 = parameters["L2 regularization param"]
    sample = parameters["sample"]
    name = "train_data_a_$(a)_D_$(D)_sample_$(sample)"
    name *= "_$(Ntr)_points"
    path = "./test_train_data/" * name * ".jld"
    data = jldopen(path, "r") do file
        read(file, name)
    end
    return data
end 

