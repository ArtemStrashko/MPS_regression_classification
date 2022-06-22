using ITensors
# using Distributed
using Random
# import Distributions: Bernoulli, Normal # MvNormal, Normal, Uniform, 
# using Dates
using Folds
using LinearAlgebra
using HDF5, JLD
using ITensors.HDF5
BLAS.set_num_threads(4)
ITensors.Strided.disable_threads()
#using MKL
# Random.seed!(1234)


function getParameters()
    
    """
    Setting up model parameters. 
    """
    
    parameters = Dict("mps_bond_dim" => 5, # initial MPS bond dim
                      "MPO_bond_dim" => 4, # initial MPO bond dim
                      "spacing" => 0, # number of MPO tensors without an outcoming leg between full MPO tensors
                      "MPO_MPS_connection_dim" => 2, # (initial) dimension of indices connecting MPS and MPO
                      "downsampling" => true, # Convert 28x28 image into 14x14
                      "feature_dim" => 2, # can choose, e.g., 6 for RGB images, no restrictions:)
                      "phys_indices" => [], # initialize empty list of physical indices
                      "num_of_image_classes" => 10, # inferred from the data
                      "precision" => Float32 # or Float64
                        )
    
    optimization = Dict("num_cdg_steps" => 5, # number of optim iterations for a particular tensor update
                        "number_of_sweeps" => 1, # number of full (from left to right and from right to left) sweeps
                        "optimizer" => ConjugateGradient, # GradientDescent, ConjugateGradient
                        "num_of_epochs" => 10, 
                        "early_stopping" => false,
#                         "alpha" => 1e-3, # 0.001 Adam optimization param
#                         "beta_1" => 0.9, # 0.9 Adam optimization param
#                         "beta_2" => 0.999, # 0.999 Adam optimization param
#                         "eps_adam" => 1e-8, # 1e-8 Adam optimization param
#                         "Adam_steps" => 100 # Adam optimization param
                        )
    
    training_set = Dict("training_set_size" => 50000, 
                        "batch_size" => 50000, # 512, 256, 128, 64, 32
                        "label_noise" => 0.0 )
    
    merge!(parameters, optimization)
    merge!(parameters, training_set)
    
    return parameters
end


function setPhysAndConnectInds(parameters::Dict=parameters)
    
    """
    Sets up physical and connection indices, their positions and also 
    a label index. 
    """
    
    num_of_phys_ind = 28^2
    if parameters["downsampling"]
        num_of_phys_ind = 14^2
    end

    # get physical and connecting MPO-MPS indices
    # and initialize MPO and MPS
    feature_map_dim = parameters["feature_dim"]
    connect_dim = parameters["MPO_MPS_connection_dim"]
    phys_dim = num_of_phys_ind
    phys_ind_positions = [i for i in 1:num_of_phys_ind]
    connect_ind_positions = []
    i = 1
    while i <= num_of_phys_ind
        append!(connect_ind_positions, i)
        i += parameters["spacing"] + 1
    end

    phys_indices = [Index(feature_map_dim) for _ in 1:phys_dim] 
    connect_indices = [Index(connect_dim) for _ in 1:length(connect_ind_positions)]

    parameters["phys_indices"] = phys_indices
    parameters["connect_indices"] = connect_indices
    parameters["phys_ind_positions"] = phys_ind_positions
    parameters["connect_ind_positions"] = connect_ind_positions
    parameters["num_of_phys_ind"] = num_of_phys_ind
    parameters["label_index"] = Index(parameters["num_of_image_classes"])
end


# set up MPO and MPS tensors
function getInit_MPO_MPS_tensors(init_MPO_num, 
                                 parameters::Dict=parameters)
    
    """
    Gives MPS and MPO tensors using init_MPO_num as a parameter for MPO. 
    """
    
#     Random.seed!(1234)
    
    chi_mpo = parameters["MPO_bond_dim"]
    chi_mps = parameters["mps_bond_dim"]
    connect_ind_positions = parameters["connect_ind_positions"][:]
    phys_indices = parameters["phys_indices"]
    connect_indices = parameters["connect_indices"]
    phys_dim = length(phys_indices)
    label_index = parameters["label_index"]
    
    MPO_tensors = Array{ITensor}(undef, phys_dim)
    MPS_tensors = Array{ITensor}(undef, length(connect_ind_positions))

    # first MPS and MPO tensor
    mpo_idx_right = Index(chi_mpo)
    mps_idx_right = Index(chi_mps)
    MPS_tensors[1] = randomITensor(parameters["precision"], connect_indices[1], mps_idx_right)
    MPO_tensors[1] = firstMPO(connect_indices[1], phys_indices[1], mpo_idx_right)
    popfirst!(connect_ind_positions)

    # middle tensors
    j = 2
    for i in 2:phys_dim-1

        mpo_idx_left = mpo_idx_right
        mpo_idx_right = Index(chi_mpo)

        if length(connect_ind_positions) > 1 && i == connect_ind_positions[1]
            # MPS tensor is present
            mps_index_left = mps_idx_right
            mps_idx_right = Index(chi_mps)
            MPS_tensors[j] = tensThree(mps_index_left, connect_indices[j], mps_idx_right)
            MPO_tensors[i] = tensFour(init_MPO_num, 
                                        mpo_idx_left, connect_indices[j], 
                                        phys_indices[i], mpo_idx_right)

            j += 1 # next MPS tensor 
            popfirst!(connect_ind_positions)
        elseif length(connect_ind_positions) == 1 && i == connect_ind_positions[1]
            # last MPS tensor
            MPS_tensors[j] = randomITensor(parameters["precision"], mps_idx_right, connect_indices[j], label_index)
            MPO_tensors[i] = tensFour(init_MPO_num, 
                                        mpo_idx_left, connect_indices[j], 
                                        phys_indices[i], mpo_idx_right)
            popfirst!(connect_ind_positions)
        else
            # only in MPO index is present
            MPO_tensors[i] = tensThree(mpo_idx_left, phys_indices[i], mpo_idx_right)
        end
    end

    # last MPO (and maybe MPS) tensor
    if length(connect_ind_positions) == 1
        # one MPS tensor and MPO tensors left
        MPS_tensors[end] = randomITensor(parameters["precision"], mps_idx_right, connect_indices[end], label_index)
        MPO_tensors[end] = lastMPOConnected(mpo_idx_right, connect_indices[end], phys_indices[end])
    elseif length(connect_ind_positions) == 0
        # only MPO tensor is present
        MPO_tensors[end] = lastMPOUnconnected(mpo_idx_right, phys_indices[end])
    else
        throw(DomainError(connect_ind_positions, "More than 1 connect index left, WRONG!"))
    end
    
    return MPS(MPS_tensors), MPO(MPO_tensors)
end


# function firstMPS(bot_idx, right_idx, parameters=parameters)
#     tens = ITensor(parameters["precision"], bot_idx, right_idx)
#     for i in 1:min(dim(bot_idx), dim(right_idx))
#         tens[bot_idx => i, right_idx => i] = 1.0
#     end
# end


function firstMPO(top_idx, bot_idx, right_idx, parameters=parameters)
    tens = randomITensor(parameters["precision"], top_idx, bot_idx, right_idx) * 1e-2
#     tens = ITensor(parameters["precision"], top_idx, bot_idx, right_idx)
    for i in 1:dim(right_idx)
        for j in 1:min(dim(top_idx), dim(bot_idx))
            tens[top_idx => j, bot_idx => j, right_idx => i] = 1.0
        end
    end
    return tens
end


function lastMPOConnected(right_idx, top_idx, bot_idx, parameters=parameters)
    tens = randomITensor(parameters["precision"], right_idx, top_idx, bot_idx) * 1e-2
#     tens = ITensor(parameters["precision"], right_idx, top_idx, bot_idx)
    for i in 1:dim(right_idx)
        for j in 1:min(dim(top_idx), dim(bot_idx))
            tens[right_idx => i, top_idx => j, bot_idx => j] = 1.0
        end
    end
    return tens
end


function lastMPOUnconnected(right_idx, bot_idx, parameters=parameters)
    tens = randomITensor(parameters["precision"], right_idx, bot_idx) * 1e-2
#     tens = ITensor(parameters["precision"], right_idx, bot_idx)
    for i in 1:min(dim(right_idx), dim(bot_idx))
        tens[right_idx => i, bot_idx => i] = 1.0
    end
    return tens
end


function tensThree(left_idx::Index, mid_idx::Index, right_idx::Index, parameters=parameters)
    """
    Helper function to set up MPS tensors "in the bulk"
    """
#     Random.seed!(1234)
    tens = randomITensor(parameters["precision"], left_idx, mid_idx, right_idx) * 1e-2
#     tens = ITensor(parameters["precision"], left_idx, mid_idx, right_idx)
    for i in 1:dim(left_idx)
        for j in 1:dim(mid_idx)
            tens[left_idx => i, mid_idx => j, right_idx => i] = 1.0 
        end
    end
    return tens
end

function tensFour(init_MPO_num, 
                   left_idx::Index, 
                   top_idx::Index, 
                   bot_idx::Index, 
                   right_idx::Index, parameters=parameters)
    """
    Helper function to set up MPO tensors "in the bulk"
    """
#     Random.seed!(1234)
    tens = randomITensor(parameters["precision"], left_idx, top_idx, bot_idx, right_idx) * 1e-2
#     tens = ITensor(parameters["precision"], left_idx, top_idx, bot_idx, right_idx)
    for i in 1:dim(left_idx)
        for j in 1:min(dim(top_idx), dim(bot_idx))
            tens[left_idx => i, top_idx => j, bot_idx => j, right_idx => i] = init_MPO_num
        end
    end
    return tens
end


function getReasonableInitConditions(train_images::Matrix{ITensor}, parameters::Dict=parameters)
    
    """
    Generates reasonable MPS and MPO initial tensors by binary searching 
    MPO initialization parameter duch that label output elements are not 
    too large or too small. 
    """

    left, right = 0.5, 1.0
    min_y_hat, max_y_hat = 0.0, Inf
    mps, mpo = getInit_MPO_MPS_tensors(0.5)
    while right - left > 1e-7 
        mid = (right + left) / 2
        mps, mpo = getInit_MPO_MPS_tensors(mid)
        min_y_hat, max_y_hat = yHatRange(mps, mpo, train_images)
    
        if min_y_hat > 1e-4 && max_y_hat < 1e+4
            return mps, mpo
        elseif min_y_hat < 1e-4
            left = mid
        elseif max_y_hat > 1e+4
            right = mid
        end
    end
    
    if min_y_hat > 1e-5 && max_y_hat < 1e+5
        # not the best, but still OK
        println("Not the best initialization, min_y_hat = $(min_y_hat), max_y_hat = $(max_y_hat)")
        return mps, mpo
    else
        throw(OverflowError(" Binary search didn't result in acceptable MPO/MPS initialization.
                Vector returned by contracting TN with images is too small
                or too larger (min_y_hat = $(min_y_hat), max_y_hat = $(max_y_hat)).
                
                TO THE FUTURE MYSELF!!!
                If spacing is nonzero, MPO bond dimension must be greater than 1 !!!
                Otherwise, you are probably getting min_y_hat = 0, max_y_hat = 0, 
                aren't you?
                
                This initialization routine works well for the feature map (sin x, cos x),
                but not that great for, e.g. (1, x). 
                
                Most likely, if you are fine with 
                min_y_hat = $(min_y_hat), max_y_hat = $(max_y_hat), 
                you can just loosen bounds on acceptable min_y_hat and max_y_hat.
                
                If not, try tuning parameters, e.g. changing MPS/MPO bond dimension
                and connection index dimension."))
    end   
end
        
        
function yHatRange(mps, mpo, images::Matrix{ITensor})
    """
    Helper function for choosing MPO initialization parameter. 
    It calculates resulting vector norms when contracting TN (MPS*MPO) with 
    all training images. 
    """
    f = x -> norm(fullContraction(mpo, mps, x))
    y_hat_min = Folds.minimum(f, collect(eachrow(images))) # , DistributedEx()
    y_hat_max = Folds.maximum(f, collect(eachrow(images))) # , DistributedEx()
    return y_hat_min, y_hat_max
end


function getBottomMPS(mpo, image, parameters::Dict=parameters)
    """
    Returns a "bottom MPS" by contracting MPO tensors with given image tensors.
    MPOs which are not contracted with MPS by means of connection indices are 
    contracted with connected MPOs, then the final bottom MPS length is given 
    by the number of connection indices, which matches top MPS length.
    
    Connected MPO absorbs disconnected onse to the righ not including the 
    next connected one. 
    
    Image can be either Vector{ITensor} or a SubArray
    """
    
    phys_dim = parameters["num_of_phys_ind"]
    spacing = parameters["spacing"]
    connect_ind_posit = parameters["connect_ind_positions"]   
 
    # contract MPO tensor with an image tensor
    bottom_MPS = Array{ITensor}(undef, phys_dim)
    for (i, (mpo_tens, data_tens)) in enumerate(zip(mpo, image))
        bottom_MPS[i] = mpo_tens * data_tens
    end
    
    # contract unconnected with MPS MPO tensors
    if spacing == 0
        return MPS(bottom_MPS)
    else
        bottom_MPS_modified = Array{ITensor}(undef, length(connect_ind_posit))
        """
        May be parallelized.
        """
        for (new_idx, i) in enumerate(connect_ind_posit)
            j = 0
            contr = 1
            while j <= spacing && j + i <= phys_dim
                contr *= bottom_MPS[i+j]
                j += 1
            end
            bottom_MPS_modified[new_idx] = contr
        end
    end
    return MPS(bottom_MPS_modified)
end  


function fullContraction(mpo, mps, 
                         image, parameters::Dict=parameters)
    """
    Returns contraction MPS * MPO * image, 
    image may be either a Vector{ITensor} or a subarray
    """
    bot_mps = getBottomMPS(mpo, image)
    return contractTwoMPS(bot_mps, mps) 
end


function contractTwoMPS(mps1, mps2)
    """
    If you keep label index on the first MPS tensor, contract from the end, 
    not from the beginning. Otherwise, cotract form the beginning (right now 
    label index is on the rightmost MPS tensor). 
    You can also move the label index whenever needed.
    """
    res = 1
    for (up, down) in zip(mps1, mps2)
        res *= down
        res *= up
    end
    return res
end


function labelToTensor(label::Int, parameters::Dict=parameters)
    """
    One-hot encoding of a label into a tensor. 
    """
    label_ind = parameters["label_index"]
    tens_label = ITensor(parameters["precision"], label_ind)
    tens_label[label + 1] = 1
    return tens_label
end


function modelProb(y::ITensor)
    """
    Transforms resulting label tensor (vector) into a normalized 
    vector of numbers deemed "probabilities". 
    """
    norm = (y*y)[]
    return [y[i]^2 / norm for i in 1:dim(y)]
end


function lossPerSample_isCorrect(image, label::Int, 
                                 mps, mpo, 
                                 parameters::Dict=parameters)
    """
    Returns loss per sample (for a given image) and also 1/0 for 
    correctly/incorrectly classified image. 
    """
    y_hat = fullContraction(mpo, mps, image)
#     y_yhat = y_hat[label + 1]
    y = labelToTensor(label)
    y_yhat = (y_hat * y)[]
    prob = y_yhat^2 / (y_hat*y_hat)[]
    
    correct = 0
    # label vs label + 1 is a source of bugs!
    model_probability = modelProb(y_hat)
    if findmax([pr for pr in model_probability])[2] == label + 1
        correct = 1
    end
    return [- log(prob), correct]
end


function Loss_and_Accuracy(mps, mpo, 
                            valid_or_test::String, parameters::Dict=parameters)
    """
    Returns loss and accuracy for a given (validation, test or train) data set. 
    valid_or_test can be either a string valid or test or train
    """
    data = loadData()
    images, labels = data[valid_or_test]
    loss, acc = Folds.reduce(+, (lossPerSample_isCorrect(image, label, mps, mpo) for 
                                    (image, label) in collect(zip(eachrow(images), labels)))) # , DistributedEx()
    loss /= length(labels)
    acc /= length(labels)
    return loss, acc
end 

function Loss_and_Accuracy_minibatch(mps, mpo, 
                                     images::Matrix{ITensor}, labels::Vector{Int64}, 
                                     parameters::Dict=parameters)
    """
    Same function as above, but aslo accepting labels and images in minibatches 
    for faster testing. 
    """
    loss, acc = Folds.reduce(+, (lossPerSample_isCorrect(image, label, mps, mpo) for 
                                    (image, label) in collect(zip(eachrow(images), labels)))) # , DistributedEx()
    loss /= length(labels)
    acc /= length(labels)
    return loss, acc
end 
    
    
function getContractions(mps, mpo, images, parameters::Dict=parameters)
    
    """
    Returns all left and right contractions. 
    Dimensions of left_contr/right_contr are (num_of_images, number_of_pixels)
    
    left_contr[i,j] is a contraction of all tensors from left to right including 
    (physical/pixel) site j for i-th image.
    
    right_contr[i,j] is a contraction of all tensors from right to left including 
    (physical/pixel) site j for i-th image.
    """
    
    phys_dim = parameters["num_of_phys_ind"]
    connect_ind_positions = parameters["connect_ind_positions"][:]
    num_images = size(images)[1]
    
    # Contract MPO with pixels tensors
    bottom_MPS = Array{ITensor}(undef, num_images, phys_dim)
    Threads.@threads for (i, image) in collect(enumerate(eachrow(images)))
        for (j, (pixel, tens)) in collect(enumerate(zip(image, mpo)))
            bottom_MPS[i, j] = tens * pixel
        end
    end

    left_contr  = Array{ITensor}(undef, num_images, phys_dim)
    right_contr = Array{ITensor}(undef, num_images, phys_dim)
    Threads.@threads for im_idx in 1:num_images
        j_left  = 1 # upper MPS left pointer
        j_right = length(connect_ind_positions) # upper MPS right pointer
        for i in 1:phys_dim
            # i is a left bottom MPS pointer
            up_left  = 1 # left upper MPS tensor
            up_right = 1 # right upper MPS tensor
            i_right  = phys_dim - i + 1 # i_right is a right bottom MPS pointer
            down_left  = bottom_MPS[im_idx, i]
            down_right = bottom_MPS[im_idx, i_right]
            
            if j_left <= length(connect_ind_positions) && i == connect_ind_positions[j_left]
                up_left = mps[j_left]
                j_left += 1
            end
            
            if j_right > 0 && i_right == connect_ind_positions[j_right]
                up_right = mps[j_right]
                j_right -= 1
            end
            
            if i == 1
                left_contr[im_idx, i] = down_left 
                left_contr[im_idx, i] *= up_left
                right_contr[im_idx, i_right] = down_right
                right_contr[im_idx, i_right] *= up_right
            else
                left_contr[im_idx, i]  = left_contr[im_idx, i - 1] * down_left 
                left_contr[im_idx, i] *= up_left
                right_contr[im_idx, i_right]  = right_contr[im_idx, i_right + 1] * down_right
                right_contr[im_idx, i_right] *= up_right
            end
        end
    end
    return left_contr, right_contr  
end


function getRanges(num_of_points::Int, batch_size::Int)
    """
    Returns a vector of tuples containing ranges for mini-batch 
    image processing. 
    """
    all_points = [i for i in 1:num_of_points]
    ranges = []
    if batch_size > num_of_points
        return [(1, num_of_points)]
    end

    while true
        if ranges == []
            push!(ranges, ((1, batch_size)))
        elseif ranges[end][2] + batch_size >= all_points[end]
            break
        else
            first = ranges[end][2] + 1
            last = first + batch_size
            push!(ranges, (first, last))
        end
    end
    
    if ranges[end][2] < all_points[end]
        push!(ranges, (ranges[end][2] + 1, all_points[end] ))
    end
    
    return ranges
end


function getAllBottomMPS(mpo, images::Matrix{ITensor}, 
                         parameters::Dict=parameters)
    """
    Returns all MPS obtained by contracting MPO with each image. 
    all_bottom_MPS[i,j] is a j-th "bottom" MPS tensor for i-th image. 
    
    Note that the number of MPS tensors for each image is given by the number 
    of connection indices, i.e. unconnected with top MPS MPO tensors are 
    contracted with connected ones (you may check out getBottomMPS(...) function). 
    """
    connect_ind_posit = parameters["connect_ind_positions"]
    num_im = size(images)[1]
    all_bottom_MPS = Array{ITensor}(undef, num_im, length(connect_ind_posit))
    Threads.@threads for (im_idx, image) in collect(enumerate(eachrow(images)))
        # if getBottomMPS returned a vector of tensors, then we would just have "=", 
        # however, now it ruturns MPS, so we need to use ".="
        all_bottom_MPS[im_idx, :] .= getBottomMPS(mpo, image)
    end
    return all_bottom_MPS
end


function _updateBottomMPS(mpo, image, left, right, updates_file)
	bot_mps_tens = 1
	#println(updates_file, "left = $(left), right = $(right)")
	#flush(updates_file)
	for idx in left:right
		bot_mps_tens *= mpo[idx] * image[idx]
	end
	return bot_mps_tens
end


function updateAllBottomMPS(i_MPO, mpo, images, bottom_MPS, updates_file, parameters=parameters)
	left, right = findLeftAndRight(i_MPO, mpo)
    spacing = parameters["spacing"]
	i_MPS = mpoIndexToMpsIndex(left) # spacing == 0 ? i_MPO : div(left, spacing + 1) + 1
	#println(updates_file, "i_MPS = $(i_MPS)")
	#flush(updates_file)
	for (im_idx, image) in enumerate(eachrow(images))
		new_mps = _updateBottomMPS(mpo, image, left, right, updates_file)
		bottom_MPS[im_idx, i_MPS] = new_mps
	end
end


function findLeftAndRight(i_MPO, mpo)

    left, right = i_MPO, i_MPO
    phys_dim = length(mpo)
    if i_MPO == phys_dim && length(inds(mpo[i_MPO])) == 3
        return left, right
    end

    while left > 1 && length(dims(mpo[left])) != 4
        left -= 1
    end

    while right < phys_dim && length(dims(mpo[right + 1])) < 4
        if right + 1 < phys_dim || (right + 1 == phys_dim && length(dims(mpo[right + 1])) == 2)
            right += 1
        else
            return left, right
        end
    end

    return left, right
end


function pathName(parameters=parameters)
    """
    Path name for saving results.
    """
    path = "mps_mpo_results/"
    spacing = parameters["spacing"]
    mps_chi = parameters["mps_bond_dim"]
    mpo_chi = parameters["MPO_bond_dim"]
    path *= "spacing_$(spacing)_mps$(mps_chi)_MPO_$(mpo_chi)"
	Ntr = parameters["training_set_size"]
	label_noise = parameters["label_noise"]
	path *= "Ntr_$(Ntr)_label_noise_$(label_noise)"
    return path
end


function saveResults(mps, mpo, parameters=parameters)
    """
    Saves MPS, MPO tensors and also physical, connection and 
    label indices. 
    """
    path = pathName()
    
    # Save MPS
    for (i, mps_tens) in enumerate(mps)
        mps_name = "mps_$(i)"
        file = h5open(path * "/" * mps_name * ".h5", "w")
        write(file, mps_name, mps_tens)
        close(file)
    end
    
    # Save MPO
    for (i, mpo_tens) in enumerate(mpo)
        mpo_name = "mpo_$(i)"
        file = h5open(path * "/" * mpo_name * ".h5", "w")
        write(file, mpo_name, mpo_tens)
        close(file)
    end
    
    # Save phys indices
    for (i, phys_index) in enumerate(parameters["phys_indices"])
        phys_ind_name = "phys_ind_$(i)"
        file = h5open(path * "/" * phys_ind_name * ".h5", "w")
        write(file, phys_ind_name, phys_index)
        close(file)
    end
    
    # Save connection indices
    for (i, connect_ind) in enumerate(parameters["connect_indices"])
        connect_ind_name = "connect_ind_$(i)"
        file = h5open(path * "/" * connect_ind_name * ".h5", "w")
        write(file, connect_ind_name, connect_ind)
        close(file)
    end
    
    # Save label index
    label_ind = parameters["label_index"]
    label_ind_name = "label_ind"
    file = h5open(path * "/" * "label_ind" * ".h5", "w")
    write(file, label_ind_name, label_ind)
    close(file)
    
end


function saveSingleMpsTensor(mps_tens, mps_index, parameters=parameters)
    path = pathName()
    mps_name = "mps_$(mps_index)"
    file = h5open(path * "/" * mps_name * ".h5", "w")
    write(file, mps_name, mps_tens)
    close(file)
end


function saveSingleMpoTensor(mpo_tens, mpo_index, parameters=parameters)
    path = pathName()
    mpo_name = "mpo_$(mpo_index)"
    file = h5open(path * "/" * mpo_name * ".h5", "w")
    write(file, mpo_name, mpo_tens)
    close(file)
end


function loadMPSandMPO(parameters=parameters)
    """
    Loads MPS, MPO tensors and all indices required. 
    """
    path = pathName()
    spacing = parameters["spacing"]
    phys_dim = parameters["num_of_phys_ind"]
    connect_ind_positions = parameters["connect_ind_positions"]
    mpo = Array{ITensor}(undef, phys_dim)
    phys_indices = Array{Index}(undef, phys_dim)
    mps = Array{ITensor}(undef, length(connect_ind_positions))
    connect_indices = Array{Index}(undef, length(connect_ind_positions))
    
    for i in 1:phys_dim
        mpo_name = "mpo_$(i)"
        file = h5open(path * "/" * mpo_name * ".h5", "r")
        mpo[i] = read(file, mpo_name, ITensor)
        close(file)
        
        phys_ind_name = "phys_ind_$(i)"
        file = h5open(path * "/" * phys_ind_name * ".h5", "r")
        phys_indices[i] = read(file, phys_ind_name, Index)
        close(file)
    end
    parameters["phys_indices"] = phys_indices
    
    for i in 1:length(connect_ind_positions)
        mps_name = "mps_$(i)"
        file = h5open(path * "/" * mps_name * ".h5", "r")
        mps[i] = read(file, mps_name, ITensor)
        close(file)
        
        connect_ind_name = "connect_ind_$(i)"
        file = h5open(path * "/" * connect_ind_name * ".h5", "r")
        connect_indices[i] = read(file, connect_ind_name, Index)
        close(file)
    end
    parameters["connect_indices"] = connect_indices
    
    # Load label index, update dict
    label_ind_name = "label_ind"
    file = h5open(path * "/" * "label_ind" * ".h5", "r")
    label_idx = read(file, label_ind_name, Index)
    close(file)
    parameters["label_index"] = label_idx
    
    return MPS(mps), MPO(mpo)
end
    

function addNoise(mps, mps_noise, mpo, mpo_noise)
    
    """
    If you believe you got stuck in a local minimum or whatever, 
    you can try to push the solution away adding noise by using 
    this function. Was not particularly useful. 
    """

    for n in 2:size(mps)[1]-1
        # add nise to tensor in the bulk
        d1, d2, d3 = dims(mps[n])
        for i in 1:d1
            for j in 1:d2
                for k in 1:d3
                    mps[n][i,j,k] *= 1 + rand() * mps_noise
                end
            end
        end
    end
    
#     for (i, _) in enumerate(mpo)
#         mpo[i] *= 1 + rand() * mpo_noise
#     end 
    return mps, mpo
end


function mpoIndexToMpsIndex(idx, parameters=parameters)
    """
    Maps MPO to MPS index. 
    Useful at finite spacing. 
    """
    spacing = parameters["spacing"]
    if spacing == 0
        return idx
    else
        return div(idx, spacing + 1) + 1
    end
end


function mpsTOmpoIndex(i_mps, parameters=parameters)
    spacing = parameters["spacing"]
    return i_mps * (spacing + 1) - spacing
end


using ITensors.NDTensors
function divide(tens, x)
    """
    Divides the each MPS/MPO tensor by a number. 
    """
    for t in tens
        t = tensor(t)
        for i in eachindex(t)
            t[i] = t[i] / x
        end
    end
end


function rescaleTN(mps, mpo, images, updates_file, parameters=parameters)
    """
    Rescales MPS and MPO tensors if label output is too large/small. 
    """
    spacing = parameters["spacing"]
    file_name = filename()
    #updates_file = open(file_name, "a")
    min_out, max_out = yHatRange(mps, mpo, images)
    if max_out > 1e+5
        x = max_out / 1e+4
        println(updates_file, "x_max = $(max_out), rescale!")
    elseif min_out < 1e-5
        x = min_out / 1e-5
        println(updates_file, "x_min = $(min_out), rescale!")
    else
        println(updates_file, "No need to rescale")
        flush(updates_file)
#         close(updates_file)
        return false # no need to rescale
    end
    
    l = length(parameters["connect_indices"]) + parameters["num_of_phys_ind"]
    x = Float32(x^(1/l))
    divide(mps, x)
    divide(mpo, x)
	saveResults(mps, mpo)
    flush(updates_file)
    #close(updates_file)
    return true
end





function getL2contractions(mps, mpo, parameters=parameters)
    
    """
    Returns contractions required for efficient L2 penalty implementations. 
    Still pretty expensive, so other simplified penalty strategies my be 
    more useful. 
    """
    
    phys_dim = parameters["num_of_phys_ind"]
    connect_ind_positions_left = parameters["connect_ind_positions"][:]
    connect_ind_positions_right = reverse(connect_ind_positions_left)
    phys_indices = parameters["phys_indices"]
    label_idx = parameters["label_index"]
    left_contr  = Array{ITensor}(undef, phys_dim)
    right_contr = Array{ITensor}(undef, phys_dim)
    
    left = 1
    right = 1
    for i in 1:phys_dim
        
        left_top = mpo[i]
        if length(connect_ind_positions_left) > 0 && i == connect_ind_positions_left[1]
            left_top *= mps[i]
            popfirst!(connect_ind_positions_left)
        end
        left_bot = noprime(prime(left_top), prime(phys_indices[i]))
        left *= left_top * left_bot
        left_contr[i] = left
        
        j = phys_dim - i + 1
        right_top = mpo[j]
        if length(connect_ind_positions_right) > 0 && j == connect_ind_positions_right[1]
            right_top *= mps[j]
            popfirst!(connect_ind_positions_right)
        end
        right_bot = noprime(prime(right_top), prime(phys_indices[j]))
        if i == 1
            noprime!(right_bot, prime(label_idx))
        end
        right *= right_top * right_bot
        right_contr[j] = right
    end
    
    return left_contr, right_contr
end


function TNsquared(mps, mpo, parameters=parameters)
    """
    Squared TN. 
    """
    l, r = getL2contractions(mps, mpo)
    return (l[12] * r[13])[]
end


function figname()
    """
    Generates figure name.
    """
    name = "acc_"
    spacing = parameters["spacing"]
    mps_chi = parameters["mps_bond_dim"]
    mpo_chi = parameters["MPO_bond_dim"]
    mpo_frozen = parameters["freeze_mpo"]
    name *= "spacing_$(spacing)_mps$(mps_chi)_MPO$(mpo_chi)_mpoFrozen_$(mpo_frozen)"
    name *= ".pdf"
    return name
end
