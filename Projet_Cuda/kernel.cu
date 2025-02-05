
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <intrin.h>
#include <omp.h>

#define ALIGNMENT	16			// Alignement des données en mémoire
#define	NB_POLYS	20000000	// Nombre de polynomes à résoudre

// Macro pour tester les codes d'erreur des fonctions Cuda
#define cudaCheckError(code,mess)	if (code != cudaSuccess) printf("Cuda erreur (%s): %s\n", mess, cudaGetErrorString(code))

typedef struct {
	float reel;
	float imaginaire;
} complexe_t;	// Type pour les complexes

// Les trois tableaux suivants contiennent les coefficients des polynômes.
float __declspec(align(ALIGNMENT)) coefs_A[NB_POLYS], coefs_B[NB_POLYS], coefs_C[NB_POLYS];
// Les deux tableaux suivants contiennent les solutions des polynômes précédents
complexe_t __declspec(align(ALIGNMENT)) Solutions_1[NB_POLYS], Solutions_2[NB_POLYS];

unsigned int maxThreadsPerBlock = 1;	// Nombre maximal de threads par bloc pour le GPU
unsigned int countMultiProcessor = 1;	// Nombre de multi-processeurs du GPU

// La fonction suivante affiche et met à jour certaines propriétés du GPU
void print_cuda_properties(void)
{
	int nb_devices;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&nb_devices);
	if (nb_devices == 0) return;

	cudaGetDeviceProperties(&prop, 0);
	printf("%s - Version %2d.%2d\n", prop.name, prop.major, prop.minor);
	maxThreadsPerBlock = prop.maxThreadsPerBlock;
	countMultiProcessor = prop.multiProcessorCount;
}

// La fonction suivante initialise les tableaux coefs_A, coefs_B et coefs_C
void init_poly_coefs(void)
{
	int i;
	for (i = 0; i < NB_POLYS; i++)
	{
		do { coefs_A[i] = ((float)rand() / RAND_MAX) * 20.0 - 10.0; } while (coefs_A[i] == 0.0);
		coefs_B[i] = ((float)rand() / RAND_MAX) * 20.0 - 10.0;
		coefs_C[i] = ((float)rand() / RAND_MAX) * 20.0 - 10.0;
	}
}

// La fonction suivante met à zéro les solutions
void raz_solutions(void)
{
	int i;
	for (i = 0; i < NB_POLYS; i++)
	{
		Solutions_1[i].reel = Solutions_1[i].imaginaire = 0.0;
		Solutions_2[i].reel = Solutions_2[i].imaginaire = 0.0;
	}
}	

// La fonction suivante affiche les racines des quatre derniers polynomes 
void print_results(char* mess)
{
	int i;
	for (i = NB_POLYS - 4; i < NB_POLYS; i++)
	{
		printf("%s\tPolynome n %d:\n", mess, i);
		printf("%s\t\t%2.4f . x^2 %c %2.4f . x %c %2.4f = 0.0\n", mess, coefs_A[i], (coefs_B[i] >= 0.0) ? '+' : '-', fabs(coefs_B[i]), (coefs_C[i] >= 0.0) ? '+' : '-', fabs(coefs_C[i]));
		printf("%s\t\t\tSolution 1 : %2.4f %c i . %2.4f\n", mess, Solutions_1[i].reel, (Solutions_1[i].imaginaire >= 0.0) ? '+' : '-', fabs(Solutions_1[i].imaginaire));
		printf("%s\t\t\tSolution 2 : %2.4f %c i . %2.4f\n", mess, Solutions_2[i].reel, (Solutions_2[i].imaginaire >= 0.0) ? '+' : '-', fabs(Solutions_2[i].imaginaire));
	}
}

// A compléter
void poly2_scalaire(float* A, float* B, float* C, complexe_t* Sols_1, complexe_t* Sols_2)
{
	float Delta;
	float delta;
	int i;
	for (i = 0; i < NB_POLYS; i++)
	{
		Delta = B[i] * B[i] - 4 * A[i] * C[i];
		delta = sqrtf(fmax(-Delta, Delta));		
		float denom = 1 / (2 * A[i]);
		float first_term = -B[i] * denom;
		float second_term = delta * denom;
		if (Delta > 0) {
			Sols_1[i].reel = first_term + second_term; Sols_1[i].imaginaire = 0;
			Sols_2[i].reel = first_term - second_term; Sols_2[i].imaginaire = 0;
		}
		else {	
			Sols_1[i].reel = first_term; Sols_1[i].imaginaire = second_term;
			Sols_2[i].reel = first_term; Sols_2[i].imaginaire = - second_term;
		}
	}
}

// A compléter
void poly2_scalaire_omp(float* A, float* B, float* C, complexe_t* Sols_1, complexe_t* Sols_2)
{
	#pragma omp parallel
	{
		float Delta;
		float delta;
		int i;
		#pragma omp for
		for (i = 0; i < NB_POLYS; i++)
		{
			Delta = B[i] * B[i] - 4 * A[i] * C[i];
			delta = sqrtf(fmax(-Delta, Delta));
			float denom = 1 / (2 * A[i]);
			float first_term = -B[i] * denom;
			float second_term = delta * denom;
			if (Delta > 0) {
				Sols_1[i].reel = first_term + second_term; Sols_1[i].imaginaire = 0;
				Sols_2[i].reel = first_term - second_term; Sols_2[i].imaginaire = 0;
			}
			else {
				Sols_1[i].reel = first_term; Sols_1[i].imaginaire = second_term;
				Sols_2[i].reel = first_term; Sols_2[i].imaginaire = -second_term;
			}
		}
	}
}

static inline __m128 compute_opposite(__m128 vect) {
	__m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));  // Masque avec uniquement le bit de signe à 1
	return _mm_xor_ps(vect, sign_mask);  // Inversion du bit de signe
}

// A compléter
	void poly2_sse2(float* A, float* B, float* C, complexe_t* Sols_1, complexe_t* Sols_2)
	{
		{
			__m128 zero = _mm_setzero_ps();                        // vecteur 0 pour les comparaisons Delta > 0
			__m128 one = _mm_set1_ps(1.0f);
			for (int i = 0; i < NB_POLYS; i+=4)
			{
				__m128 vectA = _mm_load_ps(&A[i]);
				__m128 vectB = _mm_load_ps(&B[i]);
				__m128 vectC = _mm_load_ps(&C[i]);
				__m128 b_squared = _mm_mul_ps(vectB, vectB);            // B[i] * B[i]
				__m128 ac_product = _mm_mul_ps(vectA, vectC);           // A[i] * C[i]
				__m128 scaled_ac = _mm_mul_ps(ac_product, _mm_set1_ps(4.0)); // 4 * A[i] * C[i]
				__m128 Delta = _mm_sub_ps(b_squared, scaled_ac);        // Delta = B[i] * B[i] - 4 * A[i] * C[i]
				__m128 denom = _mm_div_ps(one, _mm_add_ps(vectA, vectA)); // denom = 1/(2 * A)

				__m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));  // Mask avec tous les bits sauf le bit de signe
				__m128 delta = _mm_sqrt_ps(_mm_and_ps(Delta, abs_mask));				
				__m128 sign_mask = _mm_cmpge_ps(Delta, zero); // vaut 0xfffffffff si Delta >= 0

				__m128 first_term = _mm_mul_ps(compute_opposite(vectB), denom); // first_term = -B * denom
				__m128 second_term = _mm_mul_ps(denom, delta); // second_term = delta * denom

				__m128 reel1 = _mm_add_ps(first_term, _mm_and_ps(second_term, sign_mask));
				__m128 reel2 = _mm_sub_ps(first_term, _mm_and_ps(second_term, sign_mask));
				__m128 imaginaire1 = _mm_andnot_ps(sign_mask, second_term);
				__m128 imaginaire2 = compute_opposite(_mm_andnot_ps(sign_mask, second_term));

				__m128 vecSol1hi = _mm_unpackhi_ps(reel1, imaginaire1); __m128 vecSol1lo = _mm_unpacklo_ps(reel1, imaginaire1); // rearrange les vecteurs pour avoir reel | imaginaire | reel | imaginaire (MSB) et reel | imaginaire | reel | imaginaire (LSB)
				__m128 vecSol2hi = _mm_unpackhi_ps(reel2, imaginaire2); __m128 vecSol2lo = _mm_unpacklo_ps(reel2, imaginaire2);
				
				// Stocke 16 octets (2 complexes) dans Sols_1[i] et Sols_1[i+1]
				_mm_store_ps((float*)&Sols_1[i], vecSol1lo);
				// Stocke 16 octets (2 complexes) dans Sols_1[i+2] et Sols_1[i+3]
				_mm_store_ps((float*)&Sols_1[i + 2],vecSol1hi);
				
				_mm_store_ps((float*)&Sols_2[i], vecSol2lo);
				_mm_store_ps((float*)&Sols_2[i + 2], vecSol2hi);

			}
		}
	}

// A compléter
void poly2_sse2_omp(float* A, float* B, float* C, complexe_t* Sols_1, complexe_t* Sols_2)
{
	#pragma omp parallel
	{
		__m128 zero = _mm_setzero_ps();                        // vecteur 0 pour les comparaisons Delta > 0
		__m128 one = _mm_set1_ps(1.0f);
		#pragma omp for
		for (int i = 0; i < NB_POLYS; i += 4)
		{
			__m128 vectA = _mm_load_ps(&A[i]);
			__m128 vectB = _mm_load_ps(&B[i]);
			__m128 vectC = _mm_load_ps(&C[i]);
			__m128 b_squared = _mm_mul_ps(vectB, vectB);            // B[i] * B[i]
			__m128 ac_product = _mm_mul_ps(vectA, vectC);           // A[i] * C[i]
			__m128 scaled_ac = _mm_mul_ps(ac_product, _mm_set1_ps(4.0)); // 4 * A[i] * C[i]
			__m128 Delta = _mm_sub_ps(b_squared, scaled_ac);        // Delta = B[i] * B[i] - 4 * A[i] * C[i]
			__m128 denom = _mm_div_ps(one, _mm_add_ps(vectA, vectA)); // denom = 1/(2 * A)

			__m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));  // Mask avec tous les bits sauf le bit de signe
			__m128 delta = _mm_sqrt_ps(_mm_and_ps(Delta, abs_mask));
			__m128 sign_mask = _mm_cmpge_ps(Delta, zero); // vaut 0xfffffffff si Delta >= 0

			__m128 first_term = _mm_mul_ps(compute_opposite(vectB), denom); // first_term = -B * denom
			__m128 second_term = _mm_mul_ps(denom, delta); // second_term = delta * denom

			__m128 reel1 = _mm_add_ps(first_term, _mm_and_ps(second_term, sign_mask));
			__m128 reel2 = _mm_sub_ps(first_term, _mm_and_ps(second_term, sign_mask));
			__m128 imaginaire1 = _mm_andnot_ps(sign_mask, second_term);
			__m128 imaginaire2 = compute_opposite(_mm_andnot_ps(sign_mask, second_term));

			__m128 vecSol1hi = _mm_unpackhi_ps(reel1, imaginaire1); __m128 vecSol1lo = _mm_unpacklo_ps(reel1, imaginaire1); // rearrange les vecteurs pour avoir reel | imaginaire | reel | imaginaire (MSB) et reel | imaginaire | reel | imaginaire (LSB)
			__m128 vecSol2hi = _mm_unpackhi_ps(reel2, imaginaire2); __m128 vecSol2lo = _mm_unpacklo_ps(reel2, imaginaire2);

			// Stocke 16 octets (2 complexes) dans Sols_1[i] et Sols_1[i+1]
			_mm_store_ps((float*)&Sols_1[i], vecSol1lo);
			// Stocke 16 octets (2 complexes) dans Sols_1[i+2] et Sols_1[i+3]
			_mm_store_ps((float*)&Sols_1[i + 2], vecSol1hi);

			_mm_store_ps((float*)&Sols_2[i], vecSol2lo);
			_mm_store_ps((float*)&Sols_2[i + 2], vecSol2hi);

		}
	}
}

// A compléter
__global__ void poly2_cuda(float* A, float* B, float* C, complexe_t* Sols_1, complexe_t* Sols_2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// On ne depasse pas la taille du tableau de poly
	if (i >= NB_POLYS) return;	

	float Delta;
	float delta;

	// Chargement des coefficients dans des registres pour accès rapide
	float currentA = A[i];
	float currentB = B[i];
	float currentC = C[i];

	Delta = currentB * currentB - 4 * currentA * currentC;
	delta = sqrtf(fmax(-Delta, Delta));
	float denom = 1 / (2 * currentA);
	float first_term = -currentB * denom;
	float second_term = delta * denom;
	if (Delta > 0) {
		Sols_1[i].reel = first_term + second_term; Sols_1[i].imaginaire = 0;
		Sols_2[i].reel = first_term - second_term; Sols_2[i].imaginaire = 0;
	}
	else {
		Sols_1[i].reel = first_term; Sols_1[i].imaginaire = second_term;
		Sols_2[i].reel = first_term; Sols_2[i].imaginaire = -second_term;
	}
}



#define NUM_THREADS_PER_BLOCKS 128


int main()
{
	unsigned long long Debut, Fin, dureeScalaire, dureeScalaireOMP, dureeSSE, dureeSSEOMP, dureeCuda;
	float* ptr_Cuda_coefs_A, * ptr_Cuda_coefs_B, * ptr_Cuda_coefs_C;
	complexe_t* ptr_Cuda_sols_1, * ptr_Cuda_sols_2;
	cudaError_t cudaError;

	// on choisit un nb de block suffisant pour pouvoir parcourir l'ensemble des poly
	// ici la valeur entiere sup
	int nbBlocks = (NB_POLYS + NUM_THREADS_PER_BLOCKS - 1) / NUM_THREADS_PER_BLOCKS;

	print_cuda_properties();
	init_poly_coefs();

	Debut = __rdtsc();
	poly2_scalaire(coefs_A, coefs_B, coefs_C, Solutions_1, Solutions_2);
	Fin = __rdtsc();
	dureeScalaire = Fin - Debut;
	print_results("Scal    ");

	Debut = __rdtsc();
	poly2_scalaire_omp(coefs_A, coefs_B, coefs_C, Solutions_1, Solutions_2);
	Fin = __rdtsc();
	dureeScalaireOMP = Fin - Debut;
	print_results("Scal OMP");

	Debut = __rdtsc();
	poly2_sse2(coefs_A, coefs_B, coefs_C, Solutions_1, Solutions_2);
	Fin = __rdtsc();
	dureeSSE = Fin - Debut;
	print_results("SSE2    ");

	Debut = __rdtsc();
	poly2_sse2_omp(coefs_A, coefs_B, coefs_C, Solutions_1, Solutions_2);
	Fin = __rdtsc();
	dureeSSEOMP = Fin - Debut;
	print_results("SSE2 OMP");

	cudaCheckError(cudaMalloc(&ptr_Cuda_coefs_A, NB_POLYS * sizeof(float)), "cudaMalloc - coefs_A");
	cudaCheckError(cudaMalloc(&ptr_Cuda_coefs_B, NB_POLYS * sizeof(float)), "cudaMalloc - coefs_B");
	cudaCheckError(cudaMalloc(&ptr_Cuda_coefs_C, NB_POLYS * sizeof(float)), "cudaMalloc - coefs_C");
	cudaCheckError(cudaMalloc(&ptr_Cuda_sols_1, NB_POLYS * sizeof(complexe_t)), "cudaMalloc - sols_1");
	cudaCheckError(cudaMalloc(&ptr_Cuda_sols_2, NB_POLYS * sizeof(complexe_t)), "cudaMalloc - sols_2");
	cudaCheckError(cudaMemcpy(ptr_Cuda_coefs_A, coefs_A, NB_POLYS * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy - coefs_A");
	cudaCheckError(cudaMemcpy(ptr_Cuda_coefs_B, coefs_B, NB_POLYS * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy - coefs_B");
	cudaCheckError(cudaMemcpy(ptr_Cuda_coefs_C, coefs_C, NB_POLYS * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy - coefs_C");

	Debut = __rdtsc();
	poly2_cuda <<< nbBlocks, NUM_THREADS_PER_BLOCKS >>> (ptr_Cuda_coefs_A, ptr_Cuda_coefs_B, ptr_Cuda_coefs_C, ptr_Cuda_sols_1, ptr_Cuda_sols_2);
	cudaCheckError(cudaGetLastError(), "cuda Kernel poly2");
	cudaDeviceSynchronize();
	Fin = __rdtsc();
	dureeCuda = Fin - Debut;
		
	cudaCheckError(cudaMemcpy(Solutions_1, ptr_Cuda_sols_1, NB_POLYS * sizeof(complexe_t), cudaMemcpyDeviceToHost), "cudaMemcpy - sols_1");
	cudaCheckError(cudaMemcpy(Solutions_2, ptr_Cuda_sols_2, NB_POLYS * sizeof(complexe_t), cudaMemcpyDeviceToHost), "cudaMemcpy - sols_2");
	cudaCheckError(cudaFree(ptr_Cuda_coefs_A), "cudaFree - coefs_A");
	cudaCheckError(cudaFree(ptr_Cuda_coefs_B), "cudaFree - coefs_B");
	cudaCheckError(cudaFree(ptr_Cuda_coefs_C), "cudaFree - coefs_C");
	cudaCheckError(cudaFree(ptr_Cuda_sols_1), "cudaFree - sols_1");
	cudaCheckError(cudaFree(ptr_Cuda_sols_2), "cudaFree - sols_2");

	print_results("Cuda");

	printf("Duree scalaire       : %lld cycles\n", dureeScalaire);
	printf("Duree scalaire OMP   : %lld cycles - Gain = %2.2f\n", dureeScalaireOMP, ((double)dureeScalaire) / dureeScalaireOMP);
	printf("Duree SSE            : %lld cycles - Gain = %2.2f\n", dureeSSE, ((double)dureeScalaire) / dureeSSE);
	printf("Duree SSE OMP        : %lld cycles - Gain = %2.2f\n", dureeSSEOMP, ((double)dureeScalaire) / dureeSSEOMP);
	printf("Duree cuda           : %lld cycles - Gain = %2.2f\n", dureeCuda, ((double)dureeScalaire) / dureeCuda);

	return 0;
}
