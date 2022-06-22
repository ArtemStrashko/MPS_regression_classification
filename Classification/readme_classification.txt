This folder contains code for classification using MPS+MPO approach based on an extension of an original MPS approach developed in "Supervised Learning with Tensor Networks". Rather than a single layer MPS, this approach used an additional MPO layer, which is then contracted with an MPS. When MPO is set to an identity (with bond dimension 1), one recovers a single MPS approach. 

Files and their brief fucntionality.

1. GetingData.jl 
Used for getting data in a tensorized format, suitable for further tensor operations.

2. MPS_MPO_scripts.jl
Contains most of the functionality used for MPS/MPO setup, initialization and so on. 
Importanly, contains explanation of model and optimization parameters. 

3. Optimization.jl
Contains MPS+MPO optimization script.

4. OptimKit_correction.jl
Contains a slight correction (following authors' suggestion) to the OptimKit code. 

5. Note.pdf
Contain some brief technical details and the results of some preliminary MPS+MPO classification experiments. 

6. run_me_local.txt 
Contains a script for running code on a cluster.

7. test.jl
Main script for running code. All the classification results presented in the paper (Fig.5,9) can be reproduced by changing the following parameters:
parameters["mps_bond_dim"] (MPS bond dimension)
parameters["training_set_size"] (training set size)
parameters["label_noise"] (fraction of randomly corrupted training labels)
parameters["num_of_epochs"] (number of epochs)
Current version of the test.jl runs calculations, which resulted in Fig.9 in the paper.
