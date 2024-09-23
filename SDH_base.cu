#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;

/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/* 
    CUDA Kernel to compute pairwise distances and update the histogram 
*/
__global__ void SDH_kernel(atom *d_atom_list, bucket *d_histogram, long long PDH_acnt, double PDH_res, int num_buckets) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= PDH_acnt) return;

    for (int j = i + 1; j < PDH_acnt; j++) {
        double dist = sqrt(
            (d_atom_list[i].x_pos - d_atom_list[j].x_pos) * (d_atom_list[i].x_pos - d_atom_list[j].x_pos) +
            (d_atom_list[i].y_pos - d_atom_list[j].y_pos) * (d_atom_list[i].y_pos - d_atom_list[j].y_pos) +
            (d_atom_list[i].z_pos - d_atom_list[j].z_pos) * (d_atom_list[i].z_pos - d_atom_list[j].z_pos)
        );

        int h_pos = (int)(dist / PDH_res);
        if (h_pos < num_buckets) {
            atomicAdd((unsigned long long*)&(d_histogram[h_pos].d_cnt), 1ULL);
        }
    }
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time(int ver) {
    long sec_diff, usec_diff;
    gettimeofday(&endTime, &Idunno);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff= endTime.tv_usec - startTime.tv_usec;
    if (usec_diff < 0) {
        sec_diff--;
        usec_diff += 1000000;
    }

	if (ver == 1) 
		printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	else
		printf("Running time for GPU version: %ld.%06ld\n", sec_diff, usec_diff);

    return (double)(sec_diff * 1.0 + usec_diff / 1000000.0);
}

/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram() {
    int i;
    long long total_cnt = 0;

    for (i = 0; i < num_buckets; i++) {
        if (i % 5 == 0) /* we print 5 buckets per row */
            printf("\n%02d: ", i);
        
        printf("%15lld ", histogram[i].d_cnt); 
        total_cnt += histogram[i].d_cnt;
        
        /* we also want to make sure the total distance count is correct */
        if (i == num_buckets - 1)
            printf("\nT:%lld \n", total_cnt);
        else
            printf("| ");
    }
}

int main(int argc, char **argv) {
    int i;

    PDH_acnt = atoi(argv[1]);
    PDH_res = atof(argv[2]);
    //printf("args are %d and %f\n", PDH_acnt, PDH_res);

    num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
    histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    atom_list = (atom *)malloc(sizeof(atom) * PDH_acnt);

    srand(1);

    /* generate data following a uniform distribution */
    for (i = 0; i < PDH_acnt; i++) {
        atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    }

    /* start counting time */
    gettimeofday(&startTime, &Idunno);

    /* call CPU single thread version to compute the histogram */
    PDH_baseline();

    /* check the total running time */ 
    report_running_time(1);

    /* print out the histogram */
    printf("\nCPU Histogram:");
    output_histogram();

    /* GPU */
    // start counting time 
    gettimeofday(&startTime, &Idunno);

    // Allocate memory on device
    atom *d_atom_list;
    bucket *d_histogram;
    cudaMalloc((void**)&d_atom_list, PDH_acnt * sizeof(atom));
    cudaMalloc((void**)&d_histogram, num_buckets * sizeof(bucket));

    // Copy data to device
    cudaMemcpy(d_atom_list, atom_list, PDH_acnt * sizeof(atom), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, num_buckets * sizeof(bucket));  // Initialize histogram on GPU to zero

    // Launch the kernel
    int blockSize = 256;
    int gridSize = (PDH_acnt + blockSize - 1) / blockSize;
    SDH_kernel<<<gridSize, blockSize>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets);

    // Copy result back to host
    bucket *gpu_histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    cudaMemcpy(gpu_histogram, d_histogram, num_buckets * sizeof(bucket), cudaMemcpyDeviceToHost);
	printf("\n");
    
	report_running_time(2); // GPU running time 

    // Output the GPU histogram
    printf("\n\nGPU Histogram:");
    long long total_cnt = 0;
    for (i = 0; i < num_buckets; i++) {
        if (i % 5 == 0)  /* print 5 buckets per row */
            printf("\n%02d: ", i);
        printf("%15lld ", gpu_histogram[i].d_cnt); 
        total_cnt += gpu_histogram[i].d_cnt;
        if (i == num_buckets - 1)
            printf("\n T:%lld \n", total_cnt);
        else
            printf("| ");
    }

    // Compare CPU and GPU histograms
    printf("\n\nComparing CPU and GPU histograms (0 means no difference):");

    int num_dif = 0;
    total_cnt = 0;
    long long current_cnt;

    // Output histogram of differences between the CPU and GPU
    for (i = 0; i < num_buckets; i++) {
        if (i % 5 == 0)  /* print 5 buckets per row */
            printf("\n%02d: ", i);
        
        current_cnt = total_cnt;
        printf("%15lld ", histogram[i].d_cnt - gpu_histogram[i].d_cnt); 
        total_cnt += histogram[i].d_cnt - gpu_histogram[i].d_cnt;

        if (current_cnt != total_cnt)
            num_dif += 1;

        if (i == num_buckets - 1)
            printf("\nT:%lld \n", total_cnt);
        else
            printf("| ");
    }

    if (num_dif == 0)
        printf("\nThere are no differences.");
    else 
        printf("\nThere are %d differences.", num_dif);
    
    printf("\n\n");

    // Free device memory
    cudaFree(d_atom_list);
    cudaFree(d_histogram);

    // Free host memory
    free(histogram);
    free(atom_list);
    free(gpu_histogram);

    return 0;
}