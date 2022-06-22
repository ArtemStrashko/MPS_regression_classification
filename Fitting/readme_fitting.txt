This folder contains the code used for obtaining fitting (regression) results. Below we summarise how to reproduce our results. 

1. Go to the folder "Fitting/Inversion_and_compression". 
2. Run the following command "julia get_data.jl" (should be done once, but takes around an hour). This will generate data for most of the subsequent calculations. 
3. Create a directory "Fitting/Inversion_and_compression/Different_Ntr/txt_updates".
4. Now you can go to the folder "Fitting/Inversion_and_compression/Different_Ntr" and run "Exact_inv.jl Ntr" for different numbers of training points (where Ntr stays for the element of an array of numbers of training points, e.g., it can be anything from 0 to 16, which will resultin 50 to 850 training points). Calculations for different numnber of training set size are independent. Once calculations are finished, you will have test and training mean values and standard deviations versus MPS bond dimension. 
5. You can plot the results using Loss_vs_bond_dim_different_Ntr_inversion.ipynb and get Fig.2 of the paper. 

6. Go to the folder "Fitting/Inversion_and_compression/Different_epsilon" and run "julia Get_data_generating_MPSes.jl". This will create a new folder "Fitting/Inversion_and_compression/MPS" with data-generating MPSes with different values of a parameter epsilon controlling the effect of higher order terms. This will also create another folder "Fitting/Inversion_and_compression/Different_epsilon/test_train_data" with data-generating MPSes. 
7. Create a directory "Fitting/Inversion_and_compression/txt_updates". 
8. Now you can run "Exact_inv_different_a.jl a" (where a is a number from 0 to 9, which sets up the value of epsilon). Calculations for different values of epsilon are independent. 
9. You can plot the results using Loss_vs_bond_dim_different_a_inversion.ipynb and get Fig.3 and Fig.7 of the paper. 

10. Go to the folder "Fitting/Optimization". Create a folder "test_train_data". From "Fitting/Inversion_and_compression/test_train_data" copy all the data and MPS corresponding to a data-generating MPS bond dimension you want to use. For example, from the directory "Fitting/Optimization/test_train_data" you can run a command similar to the following: "scp -r ~/MPS_generalization_and_overfitting_bond_dimension_effect/Fitting/Inversion_and_compression/test_train_data/*a_0.3* .", which would copy all you need to run DMRG-like optimization with epsilon=0.3. 
11. Run "julia Optim.jl seed chi", where "seed" is the number of data sample and chi is an MPS bond dimension (trainable MPS, which you want to optimize). 
12. Once you finished calculations for all bond dimensions and for some of values of "seed" (for each seed calculations run for around one day), you can plot the results using Plotting_results_03.ipynb. This way you can get Fig.4,6,8.
