/*******************************************************************************/
/* This is a simple implementation of Elastic Differential Evolution (E-DE).   */
/* The codes are written in C.                                                 */
/* For any questions, please contact J. Chen (junxianchen001@gmail.com).       */
/* E-DE_1.0, Edited on August, 2019.                                           */
/*******************************************************************************/

// add your header files here 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <malloc.h>
#include <float.h>
#include "boost/random.hpp"

// change any of these parameters to match your needs 
#define PI 3.141592654
#define URAND  ((double)rand()/((double)RAND_MAX+1.0))


// declaration of functions used by this algorithm 

// 1. load and arrange the data 
double **data;
double **loadData(int *d, int *n);
void malloc1D(double *&a, int D);
void malloc1E(int *&a, int D);
void malloc2D(int **&a, int xDim, int yDim);
void malloc2E(double **&a, int xDim, int yDim);

// 2. functions to implement the E-DE algorithm 
double func(double k_num, double *p_pars, double *p_index, int *in_clu, int N, int D);
double getDistance(double *avector, double *bvector, int n);
int e_de(double *p, double *p2, int N, int D, int Gmax, double *Xl, double *Xu);

/********************************************************************
* func_name: fitness calculation
* input: the cluster number, population info, data info
* descript: calculating the fitness of an individual    
* this takes a user defined function                            
********************************************************************/

double func(double k_num, double *p_pars, double *p_index, int *in_clu, int N, int D) {
	double fes;  
	/* define your testing function here */
	return fes;
}

/*****************************************************************
* func_name: the gaussian generator
* input: mean and standard deviation
* descript: for the crossover operation
*****************************************************************/
double sampleNormal(double mean, double sigma)
{
	// apply the unix time to set the seed of rand
	static boost::mt19937 rng(static_cast< unsigned >(std::time(0)));

	// select the gaussian random distribution
	boost::normal_distribution< double > norm_dist(mean, sigma);

	// generate the random generator
	boost::variate_generator< boost::mt19937&, boost::normal_distribution< double > > normal_sampler(rng, norm_dist);

	return normal_sampler();
}


/*****************************************************************
* func_name: load the testing data as you need 
* input: dimension and data number
* descript: load data from the txt file
*****************************************************************/
double **loadData(int *d, int *n)
{
	int i, j;
	double **arraydata; 
	FILE *fp;

	if ((fp = fopen("testing_data.txt", "r")) == NULL)    
			fprintf(stderr, "cannot open data.txt!\n");
	
	if (fscanf(fp, "D=%d,N=%d\n", d, n) != 2)   fprintf(stderr, "load error!\n");
	malloc2E(arraydata, *n, *d); 

	for (i = 0; i<*n; i++)
		for (j = 0; j<*d; j++)
			fscanf(fp, "%lf", &arraydata[i][j]); 

	return arraydata;
}

/*****************************************************************
* func_name: array function
* input: array name and dimension
* descript: allocate the int or double array
*****************************************************************/
void malloc1D(double *&a, int D) {
	a = (double *)malloc(D * sizeof(double));
	if (a == NULL)
		perror("malloc");
}

void malloc1E(int *&a, int D) {
	a = (int *)malloc(D * sizeof(int));
	if (a == NULL)
		perror("malloc");
}

void malloc2D(int **&a, int xDim, int yDim)
{
	a = (int **)malloc(xDim * sizeof(int *));
	a[0] = (int *)malloc(xDim * yDim * sizeof(int));
	for (int i = 1; i<xDim; i++)
	{
		a[i] = a[i - 1] + yDim;
	}
	if (a == NULL)
		perror("malloc");
}

void malloc2E(double **&a, int xDim, int yDim)
{
	a = (double **)malloc(xDim * sizeof(double *));
	a[0] = (double *)malloc(xDim * yDim * sizeof(double));
	for (int i = 1; i<xDim; i++)
	{
		a[i] = a[i - 1] + yDim;
	}
	if (a == NULL)
		perror("malloc");
}

/*****************************************************************
* func_name: distance calculation
* input: the two testing vertors and the dimension length
* descript: for calculating the Euclidean distance
*****************************************************************/
double getDistance(double *avector, double *bvector, int n)
{
	int i;
	double sum = 0;
	for (i = 0; i<n; i++)
		sum = sum + pow(*(avector + i) - *(bvector + i), 2);

	return sqrt(sum);
}

/*****************************************************************
* func_name: E-DE algorithm
* input: the evolution population, the data number N, dimension D
*        the lower and upper bound Xl, Xu
* descript: do the mutation and crossover opertors, and select the
*           best results for the next generation
*****************************************************************/
int e_de(double *p, double *p2, int N, int D, int Gmax, double *Xl, double *Xu) {
	int i, k, r1, r2, r3, r4, r1_full, r3_full;
	int *r1_rand, *r2_rand, *r3_rand, **in_cluster2;
	int NP = 10 * D, numofE = 0, index = 0, swap, swap1, swap_if, cr_lenth, cr_n;

	double F = 0.5, CR = 0.4, MU, gauss_index;
	double **next_index, **next_param, distance;
	double **out_area, **in_area, *dist, *center, best_val = DBL_MIN, best_val2 = 0, min;

	malloc1D(dist, D);
	malloc1D(center, D);
	malloc1E(r1_rand, 20);   
	malloc1E(r2_rand, 20);
	malloc1E(r3_rand, 20);

	malloc2E(out_area, NP, 20 * D);   
	malloc2E(in_area, NP, 20 * D);     
	malloc2E(next_index, NP, 20 * D);
	malloc2E(next_param, NP, 2);
	malloc2D(in_cluster2, NP, N);

	for (i = 0; i<NP; i++) {
		for (int j = 0; j<2; j++) {
			next_param[i][j] = 0;
		}
		for (int j = 0; j<N; j++) {
			in_cluster2[i][j] = 0;
		}
		for (int j = 0; j<20 * D; j++) {
			out_area[i][j] = 0;
			in_area[i][j] = 0;
			next_index[i][j] = 0;
		}
	}

	for (i = 0; i<D; i++) {
		dist[i] = 0;
		center[i] = 0;
	}

	for (k = 0; k<Gmax; k++)      //Gmax denotes the maximum iteration times
	{
		for (i = 0; i<NP; i++)   
		{
			//Mutation operator
			//step 1 of the mutation: random select 3 individuals
			do{
				r1 = (int)(NP*URAND);
			} while (r1 == i);
			do{
				r2 = (int)(NP*URAND);
			} while (r2 == i || r2 == r1);
			do{
				r3 = (int)(NP*URAND);
			} while (r3 == i || r3 == r1 || r3 == r2);

			//step 2 of the mutation: determine the cluster number of mutant vector
			int mutate_d1 = (int) *(p2 + r1 * 2);
			int mutate_d2 = (int) *(p2 + r2 * 2);
			int mutate_d3 = (int) *(p2 + r3 * 2);

			int mutate_u = mutate_d1 + (int)( F *(mutate_d2 - mutate_d3));

			if (mutate_u > 20)
				mutate_u = 20;
			if (mutate_u < 2)
				mutate_u = 2;

			next_param[i][0] = mutate_u;

			for (int j = 0; j<20; j++) {
				r1_rand[j] = 0;
				r2_rand[j] = 0;
				r3_rand[j] = 0;
			}
			
			//step 3 of the mutation: produce the initial mutant vector
			if (mutate_u == mutate_d1) {     
				for (int j = 0; j<mutate_d1*D; j++)
					next_index[i][j] = *(p + r1 * 20 * D + j);
			}
			else if (mutate_u > mutate_d1) {
				for (int j = 0; j<mutate_d1*D; j++)
					next_index[i][j] = *(p + r1 * 20 * D + j);

				//select from the r2 or r3 individual
				r3_full = 0;
				for (int j = mutate_d1; j<mutate_u; j++) {
					if (URAND < 0.5) {   //select from the r2
						r4 = (int)(mutate_d2*URAND);   
						while (r2_rand[r4] == 1) {   
							r4 = (int)(mutate_d2*URAND);   
						}
						for (int l = 0; l<D; l++) {
							next_index[i][j*D + l] = *(p + r2 * 20 * D + r4 * D + l);
						}
						r2_rand[r4] = 1;
					}
					else {   //select from the r3
						if (r3_full < mutate_d3) {
							r4 = (int)(mutate_d3*URAND);   
							while (r3_rand[r4] == 1) {   
								r4 = (int)(mutate_d3*URAND);   
							}
							for (int l = 0; l<D; l++) {
								next_index[i][j*D + l] = *(p + r3 * 20 * D + r4 * D + l);
							}
							r3_rand[r4] = 1;
							r3_full = r3_full + 1;
						}
						else {       
							r4 = (int)(mutate_d2*URAND);   
							while (r2_rand[r4] == 1) {   
								r4 = (int)(mutate_d2*URAND);  
							}
							for (int l = 0; l<D; l++) {
								next_index[i][j*D + l] = *(p + r2 * 20 * D + r4 * D + l);
							}
							r2_rand[r4] = 1;
						}
					}
				}
			}
			else {  //delete from the r1 individual
				int mutate_num = mutate_d1 - mutate_u;
				while (mutate_num != 0) {
					r4 = (int)(mutate_d1*URAND);
					while (r1_rand[r4] == 1) {
						r4 = (int)(mutate_d1*URAND);
					}
					r1_rand[r4] = 1;
					mutate_num = mutate_num - 1;
				}

				r1_full = 0;
				for (int j = 0; j<mutate_d1; j++) {
					if (r1_rand[j] == 1)
						continue;
					for (int l = 0; l<D; l++) {
						next_index[i][r1_full*D + l] = *(p + r1 * 20 * D + j * D + l);
					}
					r1_full = r1_full + 1;
				}
			}

			//step 4 of the mutation: fine tune the cluster centroids
			MU = 0.1 - 0.06 * k / (Gmax - 1);
			if (URAND < MU) {
				for (int j = 0; j<mutate_u; j++) {
					for (int l = 0; l<D; l++) {
						gauss_index = next_index[i][j*D + l] + sampleNormal(0,0.1)*(Xu[l] - Xl[l]);
						if (gauss_index > Xu[l]) {
							next_index[i][j*D + l] = Xu[l];
						}
						else if (gauss_index < Xl[l]) {
							next_index[i][j*D + l] = Xl[l];
						}
						else {
							next_index[i][j*D + l] = gauss_index;
						}
					}
				}
			}

			//Crossover operator
			//step 1 of the crossover: determine the length of crossover
			int rand_basic = (int)next_param[i][0];
			cr_lenth = 0;
			do {
				cr_lenth = cr_lenth + 1;
			} while (URAND < CR && cr_lenth < rand_basic);
			cr_n = (int)(rand_basic*URAND);   

			//step 2 of the crossover: determine the subspace of crossover
			int rand1;
			swap = 0, swap1 = 0;    //the number of points that outside and inside the swap area
			if ((cr_n + cr_lenth - 1) < rand_basic) {
				for (int j = cr_n; j<cr_n + cr_lenth; j++) {
					for (int l = 0; l<D; l++)
						in_area[i][(j - cr_n)*D + l] = next_index[i][j*D + l];
				}
			}
			else {
				for (int j = cr_n; j<rand_basic; j++) {
					for (int l = 0; l<D; l++)
						in_area[i][(j - cr_n)*D + l] = next_index[i][j*D + l];
				}
				for (int j = 0; j<(cr_n + cr_lenth - rand_basic); j++) {
					for (int l = 0; l<D; l++)
						in_area[i][(j + rand_basic - cr_n)*D + l] = next_index[i][j*D + l];
				}
			}

			if (cr_lenth == 1) {   
				for (int j = 0; j<20 * D; j++)
					out_area[i][j] = *(p + i * 20 * D + j);

				if ((int)*(p2 + i * 2) == 20) {    
					rand1 = (int)(((int) *(p2 + i * 2)) * URAND);
					for (int j = 0; j<D; j++)
						out_area[i][rand1*D + j] = in_area[i][j];
					for (int j = 0; j<20 * D; j++)
						next_index[i][j] = out_area[i][j];
					next_param[i][0] = *(p2 + i * 2);
				}
				else {
					for (int j = 0; j<D; j++)
						out_area[i][((int)*(p2 + i * 2)) *D + j] = in_area[i][j];
					for (int j = 0; j<20 * D; j++)
						next_index[i][j] = out_area[i][j];
					next_param[i][0] = *(p2 + i * 2) + 1;
				}
			}
			else {    
				for (int j = 0; j<D; j++)
					center[j] = 0;

				//average all node to get the swap center
				for (int j = 0; j<cr_lenth; j++) {
					for (int l = 0; l<D; l++) {
						center[l] = center[l] + in_area[i][j*D + l];
					}
				}

				for (int j = 0; j<D; j++) {
					dist[j] = fabs(in_area[i][0 * D + j] - in_area[i][(cr_lenth - 1)*D + j]) / 2; //first and last node
					center[j] = center[j] / cr_lenth;  //average of all node
				}


				for (int j = 0; j<(int) *(p2 + i * 2); j++) {
					swap_if = 0;
					for (int l = 0; l<D; l++) {
						if (fabs(*(p + i * 20 * D + j * D + l) - center[l]) > dist[l]) {   //the center can be changed
							swap_if = 1;
							break;
						}
					}
					if (swap_if == 1) {
						for (int l = 0; l<D; l++)
							out_area[i][swap1*D + l] = *(p + i * 20 * D + j * D + l);
						swap1 = swap1 + 1;
					}
				}
				swap = cr_lenth + swap1;

				//step 3 of the crossover : subarea swap
				if (swap <= 20 && swap >= 2) {
					for (int j = 0; j<cr_lenth*D; j++)
						next_index[i][j] = in_area[i][j];
					for (int j = cr_lenth * D; j<swap*D; j++)
						next_index[i][j] = out_area[i][j - cr_lenth * D];
					next_param[i][0] = swap;
				}
			}

			//Selection operator
			//step 1 of the selection: assign the data object
			for (int j = 0; j<N; j++) {
				min = DBL_MAX;    
				for (int l = 0; l<((int)next_param[i][0]); l++) {   
					distance = getDistance(data[j], &next_index[i][l*D], D);
					if (distance < min) {
						min = distance;
						in_cluster2[i][j] = l;    //record the data index
					}
				}
			}

			//step 2 of the selection: calculate the fitness
			next_param[i][1] = func(next_param[i][0], *data, next_index[i], in_cluster2[i], N, D);	
			numofE = numofE + 1;

			//step 3 of the selection: preserve the better one (depends on the testing index)
			if (next_param[i][1] > *(p2 + i * 2 + 1)) {  
				for (int j = 0; j<20 * D; j++) {
					*(p + i * 20 * D + j) = next_index[i][j];
				}
				*(p2 + i * 2 + 0) = next_param[i][0];
				*(p2 + i * 2 + 1) = next_param[i][1];
			}

			if (*(p2 + i * 2 + 1) > best_val) {     
				best_val = *(p2 + i * 2 + 1);
				best_val2 = *(p2 + i * 2 + 0);
				index = i;
			}
		}
	}
	return index;
}


int  main()
{
	srand((unsigned int)(time(NULL)));

	int i, j, D, N, Gmax, NP, best = 0, *popul_rand, **in_cluster;
	double data_min, data_max, min, distance, *uk, *lk, **popul_index, **popul_param;
	
	data = loadData(&D, &N);   //load data from the text file	

	NP = 10 * D;
	Gmax = 1000000/ NP;
	printf("The times of iteration(Gmax):%d\n", Gmax);

	malloc1D(uk, D);
	malloc1D(lk, D);
	malloc1E(popul_rand, N);

	malloc2D(in_cluster, NP, N);        //population index
	malloc2E(popul_index, NP, 20 * D);  //population cluster centroids
	malloc2E(popul_param, NP, 2);       //population info(cluster number, fitness value)

	for (i = 0; i < NP; i++) {
		for (j = 0; j < N; j++) {
			in_cluster[i][j] = 0;
		}
	}

	for (i = 0; i<D; i++) {
		data_min = DBL_MAX, data_max = DBL_MIN;
		for (j = 0; j<N; j++) {
			if (data[j][i] > data_max)
				data_max = data[j][i];
			if (data[j][i] < data_min)
				data_min = data[j][i];
		}
		
		uk[i] = data_max;
		lk[i] = data_min;
	}

	//population initialization
	for (i = 0; i<NP; i++) {
		int k_rand = rand() % 19 + 2;   //the cluster num is between[2,20]
		popul_param[i][0] = (double)k_rand;

		for (j = 0; j<k_rand; j++) {    
			for (int k = 0; k < D; k++) {
				popul_index[i][j*D + k] = lk[k] + URAND *(uk[k] - lk[k]);
			}		
		}

		//evaluate the fitness of the initial population
		for (int j = 0; j<N; j++) {
			min = DBL_MAX;   
			for (int l = 0; l<((int)popul_param[i][0]); l++) { 
				distance = getDistance(data[j], &popul_index[i][l*D], D);    
				if (distance < min) {
					min = distance;
					in_cluster[i][j] = l;    //record the data index
				}
			}
		}
		popul_param[i][1] = func(popul_param[i][0], *data, popul_index[i], in_cluster[i], N, D);	
	}
	
	best = e_de(*popul_index, *popul_param, N, D, Gmax, lk, uk); //run the E-DE algorithm

	//output the results as your need
	printf("E-DE run successful!\n");

	return 0;
}


