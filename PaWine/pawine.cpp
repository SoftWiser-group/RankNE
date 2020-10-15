#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>


#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0
#define NEG_SAMPLING_POWER_BAK 0.75

const int hash_table_size = 300000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;                    // Precision of float numbers

struct ClassVertex {
    double degree;
    char *name;
};

struct ClassEdge {
    int source;
    int dest;
};

char Fmatrix[MAX_STRING], Hmatrix[MAX_STRING], Bmatrix[MAX_STRING];
char network_file[MAX_STRING], embedding_file[MAX_STRING];
struct ClassVertex *vertex;
struct ClassEdge *edges;
int is_random = 0, num_threads = 1, dim = 128, num_negative = 1;
int *vertex_hash_table, *neg_table, *neg_size, *has_table, *hash_table2;
long long max_vertices;
long long edge_index;
int max_num_vertices = 1000, max_num_edges = 10000, num_vertices = 0;
long long total_samples = 1, current_sample_count = 0, num_edges = 0;
real step = 0.01;
real *F0, *H0, *B0, *sigmoid_table;
real neg_pow = NEG_SAMPLING_POWER;

int *edge_source_id, *edge_target_id;
double *edge_weight;
real epsilon = 1.0;

real lambda = 0.01, lambda_d = 0.01;

// Parameters for edge sampling
long long *alias;
double *prob;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

/* Build two hash table, table one maps each vertex name to a unique vertex id 
table two maps each edge with vertex id */
unsigned int Hash(char *key)
{
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*key)
    {
        hash = hash * seed + (*key++);
    }
    return hash % hash_table_size;
}

void InitHashTable()
{
    vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;
}

void InsertHashTable(char *key, int value)
{
    int addr = Hash(key);
    while (vertex_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
    vertex_hash_table[addr] = value;
}

int SearchHashTable(char *key)
{
    int addr = Hash(key);
    while (1)
    {
        if (vertex_hash_table[addr] == -1) return -1;
        if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr];
        addr = (addr + 1) % hash_table_size;
    }
    return -1;
}

void InsertHashTable2(char *key, int value)
{
    int addr = Hash(key);
    while (hash_table2[addr] != -1) addr = (addr + 1) % hash_table_size;
    hash_table2[addr] = value;
}

int SearchHashTable2(int source, int dest)
{
    char key4hash[50];
    sprintf(key4hash,"%lld", source*max_vertices+dest);
    int addr = Hash(key4hash);
    while (1)
    {
        if (hash_table2[addr] == -1) return -1;
        if (source == edges[hash_table2[addr]].source && dest == edges[hash_table2[addr]].dest) return hash_table2[addr];
        addr = (addr + 1) % hash_table_size;
    }
    return -1;
}

/* Add an edge to the edge set */
int AddEdge(int source, int dest)
{
    edges[edge_index].source = source;
    edges[edge_index].dest = dest;
    edge_index++;
    if (edge_index + 2 > max_num_edges)
    {
        max_num_edges += 10000;
        edges = (struct ClassEdge*)realloc(edges, max_num_edges * sizeof(struct ClassEdge));
    }
    char key4hash[50];
    sprintf(key4hash,"%lld", source*max_vertices+dest);
    InsertHashTable2(key4hash, edge_index - 1);
    return edge_index -1;
}

/* Add a vertex to the vertex set */
int AddVertex(char *name)
{
    int length = strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
    strncpy(vertex[num_vertices].name, name, length-1);
    vertex[num_vertices].degree = 0;
    num_vertices++;
    if (num_vertices + 2 >= max_num_vertices)
    {
        max_num_vertices += 1000;
        vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
    }
    InsertHashTable(name, num_vertices - 1);
    return num_vertices - 1;
}

/* Read network from the training file */
void ReadData()
{
    FILE *fin;
    char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
    int vid;
    double weight;

    fin = fopen(network_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }
    num_edges = 0;
    while (fgets(str, sizeof(str), fin)) num_edges++;
    fclose(fin);
    printf("Number of edges: %lld          \n", num_edges);

    edge_source_id = (int *)malloc(num_edges*sizeof(int));
    edge_target_id = (int *)malloc(num_edges*sizeof(int));
    edge_weight = (double *)malloc(num_edges*sizeof(double));
    if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    float max_degree = 0;
    fin = fopen(network_file, "r");
    num_vertices = 0;
    for (int k = 0; k != num_edges; k++)
    {
        fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

        if (k % 10000 == 0)
        {
            printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
            fflush(stdout);
        }

        vid = SearchHashTable(name_v1);
        if (vid == -1) vid = AddVertex(name_v1);
        vertex[vid].degree += weight;
        edge_source_id[k] = vid;
        if(vertex[vid].degree > max_degree) max_degree = vertex[vid].degree;

        vid = SearchHashTable(name_v2);
        if (vid == -1) vid = AddVertex(name_v2);
        vertex[vid].degree += weight;
        edge_target_id[k] = vid;

        edge_weight[k] = weight;
    }
    fclose(fin);
    printf("Number of vertices: %d          \n", num_vertices);
    max_degree ++;
    hash_table2 = (int*)malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) hash_table2[k] = -1;
    max_vertices = 1;
    while(max_vertices < num_vertices) max_vertices *= 10;

    fin = fopen(network_file, "r");
    for (int k = 0; k != num_edges; k++)
    {
        fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

        if (k % 10000 == 0)
        {
            printf("Again Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
            fflush(stdout);
        }

        int vid1 = SearchHashTable(name_v1);
        int vid2 = SearchHashTable(name_v2);
        AddEdge(vid1, vid2);

    }
    fclose(fin);
    printf("Loading Hashtable Finished             \n");
}

void ReadMatrix()
{
    printf("%s\n","matrix");
    FILE *fin;
    double value;
    char key4hash[50];
    fin = fopen(Fmatrix, "r");
    for(int ver = 0; ver != num_vertices; ver ++)
    {   
        sprintf(key4hash, "%d", ver);
        int lver = SearchHashTable(key4hash);
        lver = lver * dim;
        for(int c = 0; c != dim; c++)
        {
            fscanf(fin, "%lf", &value);
            F0[lver + c] = value;
        }
    }
    fclose(fin);
    printf("Fmatrix Read over\n");
    fflush(stdout);

    fin = fopen(Hmatrix, "r");
    for(int ver = 0; ver != num_vertices; ver ++)
    {   
        sprintf(key4hash, "%d", ver);
        int lver = SearchHashTable(key4hash);
        lver = lver * dim;
        for(int c = 0; c != dim; c++)
        {
            fscanf(fin, "%lf", &value);
            H0[lver + c] = value;
        }
    }
    fclose(fin);
    printf("Hmatrix Read over\n");
    fflush(stdout);


    real avg = (real)num_edges / (real)(num_vertices*num_vertices);
    for(int ver = 0; ver != num_vertices; ver++)
    {
        B0[ver] = vertex[ver].degree/num_vertices - avg;
    }

}

/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable()
{
    printf("%s\n","alias");
    alias = (long long *)malloc(num_edges*sizeof(long long));
    prob = (double *)malloc(num_edges*sizeof(double));
    if (alias == NULL || prob == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double *norm_prob = (double*)malloc(num_edges*sizeof(double));
    long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
    long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;

    for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
    for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

    for (long long k = num_edges - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }

    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob[cur_small_block] = norm_prob[cur_small_block];
        alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }

    while (num_large_block) prob[large_block[--num_large_block]] = 1;
    while (num_small_block) prob[small_block[--num_small_block]] = 1;

    free(norm_prob);
    free(small_block);
    free(large_block);
}

long long SampleAnEdge(double rand_value1, double rand_value2)
{
    long long k = (long long)num_edges * rand_value1;
    return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the F0, H0, B0 embedding */
void InitVector()
{
        printf("%s\n","vector");
    long long a, b;

    a = posix_memalign((void **)&F0, 128, (long long)num_vertices * dim * sizeof(real));
    if (F0 == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
        F0[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

    a = posix_memalign((void **)&H0, 128, (long long)num_vertices * dim * sizeof(real));
    if (H0 == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
        H0[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

    a = posix_memalign((void **)&B0, 128, (long long)num_vertices * sizeof(real));
    if (B0 == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    for(b = 0; b < num_vertices; b ++)
        B0[b] = 0.0;
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable()
{
    double sum = 0, cur_sum = 0, por = 0;
    int vid = 0;
    neg_table = (int *)malloc(neg_table_size * sizeof(int));
    for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, neg_pow);
    for (int k = 0; k != neg_table_size; k++)
    {
        if ((double)(k + 1) / neg_table_size > por)
        {
            cur_sum += pow(vertex[vid].degree, neg_pow);
            por = cur_sum / sum;
            vid++;
        }
        neg_table[k] = vid - 1;
    }
}

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
    real x;
    sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1 / (1 + exp(-x));
    }
}

real FastSigmoid(real x)
{
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
    seed = seed * 25214903917 + 11;
    return (seed >> 16) % neg_table_size;
}

void *TrainLINEThread(void *id)
{
    long long u, i, lu, li;
    long long j, lj;
    long long count = 0, last_count = 0, curedge;
    unsigned long long seed = (long long)id;
    real r_uij, r_ui, r_uj, loss_uij, loss_ui, loss_uj, gra;
    real * Gra = (real*)malloc(num_vertices * sizeof(real));

    while (1)
    {
        //judge for exit
        if (count > total_samples / num_threads + 2) break;

        if (count - last_count>10)
        {
            current_sample_count += count - last_count;
            last_count = count;
            printf("%cProgress: %.3lf%%", 13, (real)current_sample_count / (real)(total_samples + 1) * 100);
            fflush(stdout);
        }

        for(int iter_rand = 0; iter_rand != num_vertices; iter_rand ++)
        {
            
            curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
            u = edge_source_id[curedge];
            i = edge_target_id[curedge];
            lu = u * dim;
            li = i * dim;

            // NEGATIVE SAMPLING
            for(int neg_num = 0; neg_num < num_negative; neg_num++)
            {
                while(1)
                {
                    j = neg_table[Rand(seed)];
                    if(SearchHashTable2(u, j) != -1 && is_random == 1) continue;
                    r_ui = r_uj = 0;
                    lj = j * dim;
                    for(int c = 0; c != dim; c++)
                    {
                        r_ui += F0[lu + c] * H0[li + c];
                        r_uj += F0[lu + c] * H0[lj + c];
                    }
                    if(r_uj > (r_ui - epsilon))break;
                }
                lj = j * dim;

                r_uij = 0;
                r_ui = 0;
                r_uj = 0;
                loss_uij = 0;
                loss_ui = 0;
                loss_uj = 0;

                for(int c = 0; c != dim; c++)
                {
                    r_uij += F0[lu + c] * (H0[li + c] - H0[lj + c]);
                    r_ui += F0[lu + c] * H0[li + c];
                    r_uj += -F0[lu + c] * H0[lj + c];
                }
                r_uij += (B0[i] - B0[j]);
                loss_uij = -1 / (1 + exp(r_uij));
                loss_ui = -1 / (1 + exp(r_ui));
                loss_uj = -1 / (1 + exp(r_uj));

                // update F
                for(int c = 0; c != dim; c++)
                {
                    Gra[c] = edge_weight[curedge] * loss_uij * (H0[li+c] - H0[lj+c])
                             + lambda_d * edge_weight[curedge] * loss_ui * F0[li+c]
                             - lambda_d * edge_weight[curedge] * loss_uj * F0[lj+c]
                             + lambda * F0[lu+c];
                    F0[lu+c] -= step * Gra[c];
                }
                
                for(int c = 0; c != dim; c++)
                {
                    Gra[c] = edge_weight[curedge] * lambda_d *  loss_ui * F0[lu+c]
                             + lambda * F0[li+c];
                    F0[li+c] -= step * Gra[c];
                }
                for(int c = 0; c != dim; c++)
                {
                    Gra[c] = edge_weight[curedge] * lambda_d *(-loss_uj * F0[lu+c])
                             + lambda * F0[lj+c];
                    F0[lj+c] -= step * Gra[c];
                }
                
                // update F
                for(int c = 0; c != dim; c++)
                {
                    Gra[c] =  edge_weight[curedge] * loss_uij * F0[lu+c]
                            + lambda * H0[li+c];
                    H0[li+c] -= step * Gra[c];
                }
                for(int c = 0; c != dim; c++)
                {
                    Gra[c] =  edge_weight[curedge] * loss_uij * (-F0[lu+c])
                             + lambda * H0[lj+c];
                    H0[lj+c] -= step * Gra[c];
                }

                // update B
                gra =  edge_weight[curedge] * loss_uij + lambda * B0[i];
                B0[i] -= step * gra;

                gra = - edge_weight[curedge] * loss_uij + lambda * B0[j];
                B0[j] -= step * gra;
            }          
        }
        count++;
    }
    free(Gra);
    pthread_exit(NULL);
}

void Output()
{
    FILE *fo = fopen(embedding_file, "w");
    printf("1");
	fprintf(fo, "%d %d\n", num_vertices, dim);
    printf("2");
	fflush(stdout);
	for (int a = 0; a < num_vertices; a++)
    {
        fprintf(fo, "%s ", vertex[a].name);
        fflush(stdout);
        for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", F0[a * dim + b]);
        fprintf(fo, "\n");
        fflush(stdout);
    }
	printf("3");
    fclose(fo);
}

void TrainLINE() {
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    if (is_random == 1)
    {
        neg_pow = NEG_SAMPLING_POWER_BAK;
    }
    printf("--------------------------------\n");
    printf("Samples: %lld0000\n", total_samples / 10000);
    printf("Negative: %d\n", num_negative);
    printf("Dimension: %d\n", dim);
    printf("Lambda: %f\n", lambda);
    printf("Lambda_d: %f\n", lambda_d);
    printf("Step: %f\n", step);
    printf("Negchoice: %d\n", is_random);
    printf("--------------------------------\n");

    InitHashTable();
    ReadData();
    InitAliasTable();
    InitVector();
    ReadMatrix();
    InitNegTable();
    InitSigmoidTable();

    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);

    clock_t start = clock();
    printf("--------------------------------\n");
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

    Output();
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("PWINE: PAir-Wise Information Network Embedding\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse network data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the learnt embeddings\n");
        printf("\t-Fmatrix <file>\n");
        printf("\t\tUse <file> to init the F0\n");
        printf("\t-Hmatrix <file>\n");
        printf("\t\tUse <file> to init the H0\n");
        printf("\t-negchoice <int>\n");
        printf("\t\tselect the negative mode; default is 0 (random selection); 1(degree based)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet dimension of vertex embeddings; default is 128\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples per vertex; default is 1\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int> * 10 thousand; default is 1\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-lambda <float>\n");
        printf("\t\tSet the pairwise parameter lambda; default is 0.01\n");
        printf("\t-lambda_d <float>\n");
        printf("\t\tSet the pairwise parameter lambda_d; default is 0.01\n");
        printf("\t-step <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.01\n");
        printf("\nExamples:\n");
        printf("./pawine -train net.txt -output vec.txt -Fmatrix F.txt -Hmatrix H.txt -Bmatrix B.txt -negchoice 1 -size 128 -negative 1 -samples 1 -step 0.01 -lambda 0.01 -lambda_d 0.01 -threads 1\n\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-Fmatrix", argc, argv)) > 0) strcpy(Fmatrix, argv[i + 1]);
    if ((i = ArgPos((char *)"-Hmatrix", argc, argv)) > 0) strcpy(Hmatrix, argv[i + 1]);
    if ((i = ArgPos((char *)"-lambda", argc, argv)) > 0) lambda=atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-lambda_d", argc, argv)) > 0) lambda_d=atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-negchoice", argc, argv)) > 0) is_random = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-step", argc, argv)) > 0) step = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-epsilon", argc, argv)) > 0) epsilon = atof(argv[i + 1]);
    total_samples *= 100;
    vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
    edges = (struct ClassEdge *)calloc(max_num_edges, sizeof(struct ClassEdge));
    TrainLINE();
    return 0;
}
