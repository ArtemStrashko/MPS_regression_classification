function filename(parameters=parameters)
    spacing = parameters["spacing"]
    chi_mps = parameters["mps_bond_dim"]
    chi_mpo = parameters["MPO_bond_dim"]
    name  = "updates_spacing_$(spacing)"
    name *= "_mps_chi_$(chi_mps)"
    name *= "_MPO_chi_$(chi_mpo)"

    Ntr = parameters["training_set_size"]
    label_noise = parameters["label_noise"]
    name *= "_Ntr_$(Ntr)_label_noise_$(label_noise)"

    name *= ".txt"
    return name
end


function threePointName(parameters=parameters)
	spacing = parameters["spacing"]
    chi_mps = parameters["mps_bond_dim"]
    chi_mpo = parameters["MPO_bond_dim"]
    name  = "Three_Points_spacing_$(spacing)"
    name *= "_mps_chi_$(chi_mps)"
    name *= "_MPO_chi_$(chi_mpo)"

    Ntr = parameters["training_set_size"]
    label_noise = parameters["label_noise"]
    name *= "_Ntr_$(Ntr)_label_noise_$(label_noise)"

    name *= ".txt"
    return name
end


function optimizeFullTN(parameters=parameters)
    """
    Optimizes our tensor network. 
        1. Optimizes MPS
        2. Optimizes MPO
        3. Repeats. 
    """
    spacing = parameters["spacing"]
    file_name = filename()
    name = "updates_losses_and_acc"
    path = pathName()
    mkpath(path)
	three_point_name = threePointName()
    
    updates_file = open(file_name, "w")
    println(updates_file, "Starting optimization.")
    flush(updates_file)
    num_epochs = parameters["num_of_epochs"]
    updates_losses = Dict( "epoch_$(i)" => Dict("train_losses" => [], 
                           "train_acc" => [], "valid_loss" => [], 
                           "valid_acc" => []) for i in 1:num_epochs)
	last_valid_acc = 0.0
	best_valid_acc = 0.0
	last_train_acc = 0.0
	tr_acc = 0.0
    
    # initial MPO and MPS
    if spacing > 0 && parameters["MPO_bond_dim"] == 1
        println(updates_file, "At finite spacing, MPO bond dim must be larger than 1. Increasing it to 2.")
        flush(updates_file)
        parameters["MPO_bond_dim"] = 2
    end
    
    # Get data and generate or load previously saved MPS and MPO
    if isdir(path) && isfile(path * "/mps_1.h5")
        # Need to run it before loading data to load phys and connection indices!
        setPhysAndConnectInds() 
        mps, mpo = loadMPSandMPO()
        println(updates_file, "Loaded previously saved MPS, MPO and phys/connect/label indices")
        data = loadData()
        train_images, train_labels = data["train"]
        println(updates_file, "Loaded training data")
        flush(updates_file)
    else
        setPhysAndConnectInds() 
        data = loadData()
        train_images, train_labels = data["train"]        
        println(updates_file, "Loaded training data")
        mps, mpo = getReasonableInitConditions(train_images[1:1000,:])
        saveResults(mps, mpo)
        println(updates_file, "Generated initial MPS and MPO")
        flush(updates_file)
    end
    
    close(updates_file)
#     GC.gc()

    # process data in mini-batches
    ranges = getRanges(length(train_labels), parameters["batch_size"])
    num_of_epochs = parameters["num_of_epochs"]
    for epoch in 1:num_of_epochs
        updates_file = open(file_name, "a")
        println(updates_file, "Epoch $(epoch) out of $(num_of_epochs)")
        flush(updates_file)
        close(updates_file)
        for range in ranges
            
            updates_file = open(file_name, "a")
            println(updates_file, "\nImage range $(range)")
            flush(updates_file)
            close(updates_file)
            
            image_batch = train_images[range[1] : range[2], :]
            label_batch = train_labels[range[1] : range[2]]

            # Optimize TN            
            tr_acc = optimizeMpsMpo(mps, mpo, image_batch, label_batch, epoch, updates_losses)
            
        end
        valid_loss, valid_acc = Loss_and_Accuracy(mps, mpo, "valid")
		updates_file = open(filename(), "a")
        println(updates_file, "After $(epoch) epoch, valid loss $(valid_loss), valid accuracy $(valid_acc)")
        flush(updates_file)
        close(updates_file)
        append!(updates_losses["epoch_$(epoch)"]["valid_loss"], valid_loss)
        append!(updates_losses["epoch_$(epoch)"]["valid_acc"], valid_acc)
        jldopen(path * "/" * name * ".jld", "w") do file
            write(file, name, updates_losses)  
        end
        
        # update three point results
        last_valid_acc = valid_acc
        best_valid_acc = max(valid_acc, best_valid_acc)
        last_train_acc = tr_acc
        jldopen(path * "/" * three_point_name *".jld", "w") do file
            write(file, three_point_name, [last_valid_acc, best_valid_acc, last_train_acc])  
        end
        
#         # stop training if validation accuracy goes down
#         if epoch > 1 && parameters["early_stopping"]
#             cur_loss  = updates_losses["epoch_$(epoch)"]["valid_acc"]
#             prev_loss = updates_losses["epoch_$(epoch - 1)"]["valid_acc"]
#             if prev_loss < cur_loss
#                 return
#             end
#         end  
    end
end


function optimizeMpsMpo(mps, mpo, images, labels, epoch, updates_losses, parameters=parameters)
    
    """
    Optimizes both MPO and MPS tensors. 
    """
    
    updates_file = open(filename(), "a")
    num_cdg_steps = parameters["num_cdg_steps"]
    num_sweeps = parameters["number_of_sweeps"]
    connect_ind_positions = parameters["connect_ind_positions"]
    path = pathName()
    name = "updates_losses_and_acc"
	freeze_mpo = parameters["freeze_mpo"]

    # get contractions
    left_contr, right_contr = getContractions(mps, mpo, images)
    
    # get full bottom MPS (convenient for MPS update)
    bottom_MPS = getAllBottomMPS(mpo, images)

    # print initial train loss and accuracy
    tr_loss, tr_acc = Loss_and_Accuracy_minibatch(mps, mpo, images, labels)
    println(updates_file, "Train loss before optimization is $(tr_loss) and accuracy is $(tr_acc)")
    flush(updates_file)
    if isnan(tr_loss)
        throw(OverflowError("Calculating loss with a saved TN resulted in NaN"))
    end
    
    for sweep in 1:num_sweeps
        going_right = true
        i_MPO = 1
        i_mps = 1
        while !(i_MPO == 1 && !going_right)
            
            # optimize and update an i_mps-th MPS tensor if exists
            if i_mps <= length(connect_ind_positions) && i_MPO == connect_ind_positions[i_mps]
                lg = x -> lossGradMPSupdate_cross_entr(x, i_mps, mps, left_contr, right_contr, bottom_MPS, labels, going_right)
                mps[i_mps], fx, _ = optimize(lg, mps[i_mps], parameters["optimizer"](;maxiter = num_cdg_steps, verbosity = 0)) 
                saveSingleMpsTensor(mps[i_mps], i_mps)
            end
            
            if ! freeze_mpo
                # optimize, update an i-th MPO tensor and save it
                lg = x -> lgMPOupdate_cross_entr(x, i_MPO, mps, mpo, left_contr, right_contr, labels, images, going_right)
                mpo[i_MPO], fx, _ = optimize(lg, mpo[i_MPO], parameters["optimizer"](;maxiter = num_cdg_steps, verbosity = 0)) 
                saveSingleMpoTensor(mpo[i_MPO], i_MPO)
				updateAllBottomMPS(i_MPO, mpo, images, bottom_MPS, updates_file)
            end

            # rescale in the middle of processing
            if i_MPO == length(mpo) && rescaleTN(mps, mpo, images, updates_file)
                # update contractions
                left_contr, right_contr = getContractions(mps, mpo, images)
                bottom_MPS = getAllBottomMPS(mpo, images)
            end

            # just printing an update
            print("Processed $(i_MPO) physical site, going $(going_right ? "right" : "left")      \r")
            
            # update contraction, step and direction
            i_MPO, i_mps, going_right = contrAndStepUpdate(going_right, i_MPO, i_mps, mps, mpo, left_contr, right_contr, images)
            
        end
        
        # rescale after a full sweep
        rescaleTN(mps, mpo, images, updates_file)
        
        # print resulting train loss and accuracy
        tr_loss, tr_acc = Loss_and_Accuracy_minibatch(mps, mpo, images, labels)
        println(updates_file, "After optimization, train loss $(tr_loss), accuracy $(tr_acc)")
        flush(updates_file)
        
        append!(updates_losses["epoch_$(epoch)"]["train_losses"], tr_loss)
        append!(updates_losses["epoch_$(epoch)"]["train_acc"], tr_acc)
        jldopen(path * "/" * name * ".jld", "w") do file
            write(file, name, updates_losses)  
        end
        
    end
    close(updates_file)
	return tr_acc
end


function contrAndStepUpdate(going_right, i_mpo, i_mps, mps, mpo, left_contr, right_contr, images, parameters=parameters)
    
    """
    Updates contractions, optimization tensor step and direction after 
    i-th MPO tensor optimization. 
    """
    
    mpo_len = size(mpo)[1]
    mps_len = size(mps)[1]
    connect_ind_positions = parameters["connect_ind_positions"]
    
    if going_right
        UpdateLeftContr(i_mpo, mps, mpo, left_contr, images)
        if i_mpo < mpo_len
            if i_mps < mps_len && i_mpo == connect_ind_positions[i_mps] 
                # move MPS index to the right
                i_mps += 1
            end
            i_mpo += 1 # move MPO index to the right
        else
            if i_mpo == connect_ind_positions[i_mps]
                # reached last mps and mpo tensor, move mps index to the left
                i_mps -= 1
            end
            # move MPS index to the left
            i_mpo -= 1
            going_right = false
        end
    else # going (back) left
        UpdateRightContr(i_mpo, mps, mpo, right_contr, images)
        if i_mpo > 1
            if i_mpo == connect_ind_positions[i_mps] 
                i_mps -= 1
            end
            i_mpo -= 1
        else
            i_mpo += 1
            i_mps += 1 # there is always mps tensor on top of the first mpo tensor by construction
            going_right = true
        end
    end
    return i_mpo, i_mps, going_right
end    


function UpdateLeftContr(i, mps, mpo, left_contr, images, parameters=parameters)
    """
    Updates left contraction with a new i-th mpo (and probably mps) tensor 
    when going from left to right.
    """
    mpo_len = size(mpo)[1]
    
    # Updating left contraction (moving right). Loop over all images.
    Threads.@threads for (im_idx, image) in collect(enumerate(eachrow(images)))
        if i == 1
            left_contr[im_idx, i] = mpo[i] * image[i]
            left_contr[im_idx, i] *= mps[1]
        else
            left_contr[im_idx, i] = left_contr[im_idx, i - 1] * mpo[i] * image[i]
            if ((i < mpo_len && length(dims(mpo[i])) == 4) || 
                (i == mpo_len && length(dims(mpo[i])) == 3))
                left_contr[im_idx, i] *= mps[mpoIndexToMpsIndex(i)] 
            end
        end
    end
end 


function UpdateRightContr(i, mps, mpo, right_contr, images, parameters=parameters)
    """
    Updates right contraction with a new i-th mpo (and probably mps) tensor 
    when going from right to left.
    """
    mpo_len = size(mpo)[1]
    
    # Updating right contraction (moving left). Loop over all images.
    Threads.@threads for (im_idx, image) in collect(enumerate(eachrow(images)))
        if i == mpo_len
            if length(dims(mpo[i])) == 2
                right_contr[im_idx, i] = mpo[i] * image[i]
            else
                right_contr[im_idx, i] = mpo[i] * image[i]
                right_contr[im_idx, i] *= mps[mpoIndexToMpsIndex(i)]  
            end
        else 
            if length(dims(mpo[i])) == 3
                right_contr[im_idx, i] = mpo[i] * image[i]
                right_contr[im_idx, i] *= right_contr[im_idx, i + 1]
            else
                right_contr[im_idx, i] = mpo[i] * image[i]
                right_contr[im_idx, i] *= right_contr[im_idx, i + 1]
                right_contr[im_idx, i] *= mps[mpoIndexToMpsIndex(i)] 
            end
        end
    end
end 


function lossGradMPSupdate_cross_entr(x, i_mps, mps, left_contr, right_contr, bottom_MPS, labels, 
                                        parameters=parameters)
    
    """
    Return loss (cross entropy) and its gradient. 
    
    Input:
        1. x - top MPS tensor we want to optimize
        2. i - index of this tensor
        3. mps - all top MPS tensors
        4. left_MPS_contr - all left MPS contractions (for all images and sites)
        5. right_MPS_contr - all left MPS contractions (for all images and sites)
        6. bottom_MPS - all bottom MPS (MPO contracted with all images)
        7. labels - image labels
    """
    
    lg_per_sample = j -> lgMPS_crossEntrPerSample(j, x, i_mps, left_contr, right_contr, bottom_MPS, labels)
    # j in an image index (number of a given image)
    loss, grad = Folds.reduce(+, 
                        (lg_per_sample(j) for j in collect(1:length(labels)))) # , DistributedEx()
#     loss, grad = sum([lg_per_sample(j) for j in collect(1:length(labels))])
    num_images = length(labels)
    grad ./= num_images # ./= is essential
    
    return loss / num_images, grad 
end;


function lgMPS_crossEntrPerSample(j, x, i_mps, left_contr, right_contr, bottom_MPS, labels, 
                                    parameters=parameters)
    """
    Loss and gradient per image. 
    
    j - number of an image
    i_mps - number of an MPS tensor
    x - given i-th MPS tensor
    """
    y_hat, d_yhat_dW = getYhatAndDerivMPS(j, x, i_mps, left_contr, right_contr, bottom_MPS)
    
    y = labelToTensor(labels[j])
    y_yhat = (y * y_hat)[]
    yhat_yhat = (y_hat * y_hat)[]
    
    p = y_yhat^2 / yhat_yhat
    loss_per_sample = - log(p)
    
    # constructing gradient
    """
    Due to a weired change in datatype during broadcasting, the following code 
    dp_dW = 2 * (y_yhat * (y * d_yhat_dW) / yhat_yhat - 
             y_yhat^2 * (y_hat * d_yhat_dW) / (yhat_yhat)^2)
    grad_per_sample = - dp_dW / p
    is replaced by a more cumbersome below... 
    """
    first_half = y_yhat * (y * d_yhat_dW)
    first_half ./= yhat_yhat

    second_half = y_yhat^2 * (y_hat * d_yhat_dW)
    second_half ./= (yhat_yhat)^2

    grad_per_sample = - 2 * (first_half - second_half)
    grad_per_sample ./= p

    return [loss_per_sample, grad_per_sample]
end


function getYhatAndDerivMPS(j, x, i_mps, left_contr, right_contr, bottom_MPS, 
                            parameters=parameters)
    """
    Generates label vector and its derivative. 
    
    j - number of an image
    i_mps - number of an MPS tensor
    x - given i-th MPS tensor
    """
    len_mps = size(bottom_MPS)[2]
    i_mpo = mpsTOmpoIndex(i_mps)
    sp = parameters["spacing"]
    
    y_hat = 1
    d_yhat_dW = 1
    
    y_hat *= try left_contr[j, i_mpo - 1] catch e 1 end
    y_hat *= bottom_MPS[j, i_mps]
    y_hat *= try right_contr[j, i_mpo + 1 + sp] catch e 1 end
    
    d_yhat_dW = y_hat
    y_hat *= x 

    return y_hat, d_yhat_dW
end







function lgMPOupdate_cross_entr(x, i_mpo, mps, mpo, left_contr, right_contr, labels, images, 
                                        parameters=parameters)
    
    """
    Input:
        1. x - MPO tensor we want to optimize
        2. i_mpo - index of this tensor
        3. mps, mpo - MPS, MPO tensors
        4. left_contr, right_contr - all left/right contractions 
        5. everything else is obvious
    """
    
    lg_mpo_per_sample = j -> lossGradMPO_crossEntrPerSample(j, x, i_mpo, mps, mpo, left_contr, right_contr, labels, images)
    loss, grad = Folds.reduce(+, (lg_mpo_per_sample(j) for j in collect(1:length(labels)))) # , DistributedEx()
#     loss, grad = sum([lg_mpo_per_sample(j) for j in collect(1:length(labels))])
    num_images = length(labels)
    grad ./= num_images # ./= is essential
    
    return loss / num_images, grad
end


function lossGradMPO_crossEntrPerSample(j, x, i_mpo, mps, mpo, left_contr, right_contr, labels, images, 
                                        parameters=parameters)
    """
    i_mpo - number of an image
    i - number of an MPO tensor
    x - given i-th MPO tensor
    """
    y_hat, d_yhat_dW = getYhatAndDeriv_MPO(j, x, i_mpo, mps, mpo, left_contr, right_contr, images)
    
    y = labelToTensor(labels[j])
    y_yhat = (y * y_hat)[]
    yhat_yhat = (y_hat * y_hat)[]
    
    p = y_yhat^2 / yhat_yhat
    loss_per_sample = - log(p)
    
    # constructing gradient
    """
    Due to a weired change in datatype during broadcasting, the following code 
    is replaced by a more cumbersome below... 
    dp_dW = 2 * (y_yhat * (y * d_yhat_dW) / yhat_yhat - 
             y_yhat^2 * (y_hat * d_yhat_dW) / (yhat_yhat)^2)
    grad_per_sample = - dp_dW / p
    """
    first_half = y_yhat * (y * d_yhat_dW)
    first_half ./= yhat_yhat

    second_half = y_yhat^2 * (y_hat * d_yhat_dW)
    second_half ./= (yhat_yhat)^2

    grad_per_sample = - 2 * (first_half - second_half)
    grad_per_sample ./= p
    
    return [loss_per_sample, grad_per_sample]
end


function getYhatAndDeriv_MPO(j, x, i, mps, mpo, left_contr, right_contr, images, 
                                parameters=parameters)
    """
    j - number of an image
    i - number of an MPO tensor
    x - given i-th MPO tensor we are optimizing
    """
    
    len_mpo = size(left_contr)[2]
    y_hat = 1
    d_yhat_dW = 1
    image = images[j,:]
    y_hat *= image[i]
    
    if i == 1
        # first MPO tensor
        y_hat *= right_contr[j, i + 1]
        y_hat *= mps[mpoIndexToMpsIndex(i)]
    elseif i == len_mpo
        # last MPO tensor
        y_hat *= left_contr[j, i - 1]
        if length(dims(x)) == 2
            # no connection index -> no top MPS tensor
        else
            # there is a top MPS tensor
            y_hat *= mps[mpoIndexToMpsIndex(i)]
        end
    else
        # MPO tensor in the bulk
        y_hat *= left_contr[j, i - 1]
        if length(dims(x)) == 3
            # no connection index -> no top MPS tensor
        elseif length(dims(x)) == 4
            # there is a top MPS tensor
            y_hat *= mps[mpoIndexToMpsIndex(i)]
        end
        y_hat *= right_contr[j, i + 1]
    end
    d_yhat_dW = y_hat
    y_hat *= x 
    
    if length(dims(y_hat)) != 1
        throw(ErrorException("y_hat is not a vector, but inds(y_hat) = $(inds(y_hat)). 
                Method = $(MPO_optim_method), updating $(i) MPO tensor, 
                mpoIndexToMpsIndex($(i)) = $(mpoIndexToMpsIndex(i)),
                mpoIndexToMpsIndex($(i+1)) = $(mpoIndexToMpsIndex(i+1)),
                inds(mps[$(mpoIndexToMpsIndex(i))]) = $(inds(mps[mpoIndexToMpsIndex(i)])), 
                inds(mps[$(mpoIndexToMpsIndex(i+1))]) = $(inds(mps[mpoIndexToMpsIndex(i+1)])),
                inds(mpo[$(i)]) = $(inds(mpo[i])),
                inds(mpo[$(i+1)]) = $(inds(mpo[i+1])),
                inds(left_MPO_contr[$(j), $(i - 1)]) = $(inds(left_MPO_contr[j, i - 1])),
                inds(right_MPO_contr[$(j), $(i + 2)]) = $(inds(right_MPO_contr[j, i + 2])),
                inds(x) = $(inds(x))"))
    end
    
    return y_hat, d_yhat_dW
end
