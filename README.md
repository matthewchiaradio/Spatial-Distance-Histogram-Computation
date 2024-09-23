# Spatial-Distance-Histogram-Computation
This program compares the time of computation of the CPU and GPU when computing the spatial distance histogram (SDH) of a collection of 3D points. The inputs are ran on the CPU first and then ran on the GPU after. After these computations are run, the time taken to copmlete is stated for each and the results of the histograms are compared.

# Compiling and Running / Input
To run this program, a script file is included which will compile and run the executable. Command line arguments are used to provide input to the program. The first number entered will be the total number of data points and the second number entered is the  bucket width (w) in the histogram to be computed. For example, "runScript.sh SDH_base.cu 10000 500" compiles and executes the program with 10000 data points and a bucket width of 500.
