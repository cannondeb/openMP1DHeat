#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


double 
computeDeltaT(double stabReq, double alpha, double deltaX) {
    /* Computes the required timestep to maintain adequate numerical stability
       for the scheme, given alpha and deltaX
    */
   double deltaT = stabReq * (deltaX * deltaX) / alpha;

   return deltaT;
}


void initializeDomain(double* A, int dim, double* bcs) {
    // Initialize Top and Bottom
    int size = dim * dim;
    for (int i = 0; i < dim; i++) {
        A[i] = bcs[2]; // Top Row
        A[size - 1 - i] = bcs[3]; // Bottom Row
        //printf("%d\n", size - 1 - i);
    }
    for (int i = 0; i <= dim; i++) {
        int idx = i * dim;
        A[idx] = bcs[0]; // Left Column
        A[idx - 1] = bcs[1]; // Right Column
    }
}


double 
solve(double dx, double dt, double alpha, double initTemp, 
    double tEnd, double* bcs) {
    /* Docs... Write out final solution, return time
    */
    double tStart;
    double elapsed;
    // Neighboring Points
    int up;
    int down;
    int left;
    int right;
   
    int dim = round(1 / dx); // Domain is 1x1 in real space
    if ((abs((dim * dx) - 1)) > 1E-12) {
        // Number of points not divisible by timestep...
        return -1;
    }
    int size = dim * dim; // Size of temperature array
    int N = tEnd / dt; // Number of timesteps
    // Allocate Temperature array
    double *Tprev = calloc(size, sizeof(double));
    double *Tnew = calloc(size, sizeof(double));
    double *tmp;

    initializeDomain(Tprev, dim, bcs);
    initializeDomain(Tnew, dim, bcs);

    tStart = omp_get_wtime();
    for (int n = 0; n < N; n++ ) {
        #pragma omp parallel for default(none) shared(Tprev, Tnew, alpha, dt, dx, dim, size) private(up, down, left, right)
        for (int idx = dim; idx < size - dim; idx++) {
            if (((idx % dim) != 0) && (((idx + 1) % dim) != 0)) {
                // Not at a boundary so update Temp.
                up = idx - dim;
                down = idx + dim;
                left = idx - 1;
                right = idx + 1;

                Tnew[idx] = Tprev[idx] + ((alpha * dt / (dx*dx)) *
                    (Tprev[right] + Tprev[left] - (4*Tprev[idx]) + 
                    Tprev[up] + Tprev[down]));
            }
        }
        // Swap array pointers to copy Tnew into Tprev
        tmp = Tprev;
        Tprev = Tnew;
        Tnew = tmp;
    }
    elapsed = omp_get_wtime() - tStart;

    // Write out final timestep
    FILE* fp = fopen("T.txt", "w+");
    fprintf(fp, "%f ", Tprev[0]);
    for (int idx = 1; idx < size; idx++) {
        fprintf(fp, "%f ", Tprev[idx]);
        if (((idx + 1) % dim) == 0) {
            fprintf(fp, "\n");
        }
    }

    // Free Memory, etc.
    fclose(fp);
    free(Tprev);
    free(Tnew);

    return elapsed;
}


int main() {    
    // Coefficient of Thermal Diffusivity
    double alpha = 0.1;
    // Time to end solution
    double tEnd = 100;
    // Uniform Initial Temperature
    double initTemp = 0;
    // Ratio of c * dT / (dx)^2 for numerical stability
    double stabReq = 0.25; // Stay on the safe side of the ideal requirement of 0.5.
    // Constant Temperature BCs. (Left, Right, Top, Bottom)
    double bcs[4] = {0, 100, 100, 0};
    // dx = dy for this example
    double dx = 0.01;
    double dt = computeDeltaT(stabReq, alpha, dx);

    double t1 = solve(dx, dt, alpha, initTemp, tEnd, bcs);
    printf("%f\n", t1);
    
    return 0;
}