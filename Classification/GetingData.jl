using MLDatasets

function preprocess(image, parameters=parameters)
    
    """
    Downsamples and vectorizes a 2D image.
    """
    
    if parameters["downsampling"]
        image = downsample(image)
    end
    
    v = from_vect_to_image(image, "from_image_to_vector")
    return v
end


function downsample(image, parameters=parameters)
    
    """
    Reduces size of a 2D image by averaging over four pixel values
    """
    
    l = 14
    new_image = zeros(parameters["precision"], l, l)
    for i in 0:l-1
        for j in 0:l-1
            x, y = 1 + 2*i, 1 + 2*j
            pixel_val = sum([image[x + k,y + p] for k in 0:1 for p in 0:1]) / 4 
            new_image[i+1,j+1] = pixel_val
        end
    end
    return new_image
end


function from_vect_to_image(data, direction, parameters=parameters)
    
    """
    Reshapes 2D image into 1D vector or visa versa dependinding on 
    direction, which can be either from_vec_to_image or from_image_to_vector
    """
    
    l = 28
    if parameters["downsampling"]
        l = 14
    end
    image = zeros(parameters["precision"], l,l)
    v = zeros(parameters["precision"], l^2)
    
    if direction == "from_vec_to_image"
        if length(size(data)) == 1
            v = data
        else
            throw(DomainError(data, "must pass a vector to convert to a 2D array")) 
        end
    elseif direction == "from_image_to_vector"
        if length(size(data)) == 2
            image = data
        else
            throw(DomainError(data, "must pass a 2D array to conver to a vector"))
        end
    else
        throw(DomainError(direction, "must a string be either from_vec_to_image or from_image_to_vector"))
    end
    
    go_right = true
    s, row, col = 1, 1, 1
    while s <= l^2
        
        if direction == "from_vec_to_image"
            image[row, col] = v[s]
        elseif direction == "from_image_to_vector"
            v[s] = image[row, col]
        end
        
        if go_right && col < l
            col += 1
        elseif go_right && col == l
            row += 1
            go_right = false
        elseif ! go_right && col > 1
            col -= 1
        elseif ! go_right && col == 1
            row += 1
            go_right = true
        end
        
        s += 1
    end
    
    if direction == "from_vec_to_image"        
        return image
    elseif direction == "from_image_to_vector"
        return v
    end
end


function makeTens(x, index, parameters=parameters)
    """
    Maps pixel value to a tensor using a triginimetric feature map.
    """
    t = ITensor(parameters["precision"], index)
    t[1] = sin(pi * x / 2) 
    t[2] = cos(pi * x / 2) 
    return t
end


function loadData(parameters=parameters)
    
    """
    Loads a distionary with tensorized training, validation and test data.
    """

    train_x_original, train_y_original = MNIST.traindata() # FashionMNIST.traindata(), CIFAR10.traindata()
    test_x,  test_labels  = MNIST.testdata(); # FashionMNIST.testdata(), CIFAR10.traindata()

	Ntr = parameters["training_set_size"]
    train_x = train_x_original[:,:,1:Ntr]
    train_labels = train_y_original[1:Ntr]

	# adding noise to the training set
	label_noise = parameters["label_noise"]
	corrupted_label_poitions = rand(1:Ntr, Int(round(label_noise * Ntr)))
	for idx in corrupted_label_poitions
		train_labels[idx] = rand(0:9)
	end
    
    valid_x = train_x_original[:,:,50_001:end]
    valid_labels = train_y_original[50_001:end];

    train_x_vectors = [preprocess(train_x[:,:,i], parameters) for i in 1:size(train_x)[end]]
    valid_x_vectors = [preprocess(valid_x[:,:,i], parameters) for i in 1:size(valid_x)[end]]
    test_x_vectors  = [preprocess(test_x[:,:,i], parameters) for i in 1:size(test_x)[end]]
    
    train_x_tensors = Array{ITensor}(undef, length(train_x_vectors), length(train_x_vectors[1]))
    valid_x_tensors = Array{ITensor}(undef, length(valid_x_vectors), length(valid_x_vectors[1]))
    test_x_tensors  = Array{ITensor}(undef, length(test_x_vectors), length(test_x_vectors[1]))
    
    phys_indices = parameters["phys_indices"]
    
    Threads.@threads for (i, image) in collect(enumerate(train_x_vectors))
        for (j, (pixel_val, index)) in enumerate(zip(image, phys_indices))
            train_x_tensors[i,j] = makeTens(pixel_val, index)
        end
    end
    
    Threads.@threads for (i, image) in collect(enumerate(valid_x_vectors))
        for (j, (pixel_val, index)) in enumerate(zip(image, phys_indices))
            valid_x_tensors[i,j] = makeTens(pixel_val, index)
        end
    end
    
    Threads.@threads for (i, image) in collect(enumerate(test_x_vectors))
        for (j, (pixel_val, index)) in enumerate(zip(image, phys_indices))
            test_x_tensors[i,j] = makeTens(pixel_val, index)
        end
    end

    data = Dict("train" => (train_x_tensors, train_labels), 
                "valid" => (valid_x_tensors, valid_labels), 
                "test"  => (test_x_tensors,  test_labels))
    return data
    
end
