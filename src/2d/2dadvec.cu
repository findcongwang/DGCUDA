/* 2dadvec.cu
 * 
 * This file calls the kernels in 2dadvec_kernels.cu for the 2D advection
 * DG method.
 */

void read_mesh(char *filename) {
    FILE *mesh_file;
    int i, n_elem, n_edge;
    float *V1x, *V1y, *V2x, *V2y, *V3x, *V3y;

    mesh_file = fopen(filename, "rt");
    line = char[100];

    // first line should be the number of elements
    fgets(line, 100 ,mesh_file);
    sscanf(line, "%f", &n_elem);

    // allocate vertex points
    V1x = (float *) malloc(n_elem * sizeof(float));
    V1y = (float *) malloc(n_elem * sizeof(float));
    V2x = (float *) malloc(n_elem * sizeof(float));
    V2y = (float *) malloc(n_elem * sizeof(float));
    V3x = (float *) malloc(n_elem * sizeof(float));
    V3y = (float *) malloc(n_elem * sizeof(float));

    i = 0;
    while(fgets(line, 100, mesh_file) != NULL) {
        sscanf(line, "%f", elem[i]);
    }
}

void initGPU() {
    //cudaMalloc((void **) &d_something, n * sizeof(float));
    cudaMalloc((void **) &d_c, n_elem * n_p * sizeof(float));
}

int main() {
    return 0;
}
