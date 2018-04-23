#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>

#define NULL_VERTEX_ID -1
#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float float32;
typedef char *string; 

struct SparseMatrix{
    int rn, cn, nnz;
    int *indices, *indptr;
    float32 *data;
};

struct AliasMat{ 
    int *alias;
    float32 *prob;
};

typedef struct SparseMatrix *P2SparseMatrix;
typedef struct AliasMat *P2AliasMat;

//////
void ReadData();
void InitAlias();
void LoadGraph(string csrmat_file);
int AddVertex(char *name);

struct AliasMat *AliasMethod(SparseMatrix *smat);
int BiSample(struct SparseMatrix *smat, struct AliasMat *alsmat, int ri);

void PrintSpaMat(struct SparseMatrix *smat, int s2d);
void Update(int vs_idx, int vt_idx, float32 *vec_error, int label);

float32 FastSigmoid(float32 x);
int Rand(unsigned long long &seed);
unsigned int Hash(char *key);
void InitHashTable();
void InsertHashTable(char *key, int value);
int SearchHashTable(char *key);

void *TrainThread(void *id);
void TrainModel();
int ArgPos(char *str, int argc, char **argv);
void Output();

//////
const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

int num_threads = 10, num_negative = 5, dim = 2;
float32 init_rho = 1, rho, gam= 7, gclip=5;
char graph_file[MAX_STRING];
char embedding_file[MAX_STRING];

int num_vertices, num_edges, max_num_vertices=10000;
int *vertex_outdeg, *vertex_indeg;
float32 *vertex_weight;
P2SparseMatrix smat;
P2AliasMat alsmat;

int *vertex_hash_table, *neg_table;
string *vertex_id2name;
long long total_samples = 1, current_sample_count = 0;
float32 *emb_vertex, *emb_context, *sigmoid_table;
bool is_binary = false;

void ReadData(){

    FILE *fin;
    fin = fopen(graph_file, "rb");
    if (fin == NULL){
        printf("ERROR: file %s not found!\n", graph_file);
        exit(1);
    }

    smat = (P2SparseMatrix)calloc(1, sizeof(struct SparseMatrix));
    LoadGraph(graph_file);

    fclose(fin);
}

void InitAlias(){ alsmat = AliasMethod(smat);}

void LoadGraph(string csrmat_file){

    FILE *fin;
    int ri, ci;
    float32 w;
    int *indices, *indptr, *ccnt;
    float32 *data;
    char name_v1[MAX_STRING], name_v2[MAX_STRING], str[MAX_STRING];

    fin = fopen(csrmat_file, "rb");
    if (fin == NULL){
		printf("ERROR: csr matrix file not found!\n");
		exit(1);
	}

    vertex_outdeg = (int *)calloc(max_num_vertices, sizeof(int));
    vertex_indeg = (int *)calloc(max_num_vertices, sizeof(int));
    vertex_weight = (float32 *)calloc(max_num_vertices, sizeof(float32));
    vertex_id2name = (string *)calloc(max_num_vertices, sizeof(string *));

    if(vertex_outdeg == NULL || vertex_indeg == NULL || vertex_weight == NULL){
		printf("Error: memory allocation failed!\n");
        exit(1);
    }

    // basic statistics: vertex & edge info
	while (fgets(str, sizeof(str), fin)) num_edges++;
    fclose(fin);

    fin = fopen(csrmat_file, "rb");
    for(int i=0; i < num_edges; i++){
        fscanf(fin, "%s %s %f", name_v1, name_v2, &w);
        ri = SearchHashTable(name_v1);
        if(ri == -1) ri = AddVertex(name_v1);
        ci = SearchHashTable(name_v2);
        if(ci == -1) ci = AddVertex(name_v2);

        vertex_outdeg[ri] += 1; 
        vertex_indeg[ci] += 1; 
        vertex_weight[ri] += w; 
    }
    fclose(fin);

    printf("Number of vertices: %d, Number of edges: %d\n", num_vertices, num_edges);

    indptr = (int *)calloc((num_vertices+1), sizeof(int));
    indices = (int *)calloc(num_edges, sizeof(int));
    data = (float32 *)calloc(num_edges, sizeof(float32));
    ccnt = (int *)calloc(num_vertices, sizeof(int));

    if(indices == NULL || indptr == NULL || data == NULL || ccnt == NULL){
		printf("Error: memory allocation failed!\n");
        exit(1);
    }

    // construct sparse_matrix.indptr
    for(int i = 0, d=0; i < num_vertices; i++){
        ccnt[i] = vertex_outdeg[i];
        d += vertex_outdeg[i];
        indptr[i+1] = d;
    }

    fin = fopen(csrmat_file, "rb");
    // construct sparse_matrix.indices & data
    for(int i=0, j=0; i < num_edges; i++){
        fscanf(fin, "%s %s %f\n", name_v1, name_v2, &w);
        ri = SearchHashTable(name_v1);
        ci = SearchHashTable(name_v2);
        j = indptr[ri] + ccnt[ri] - 1;
        indices[j] = ci;
        data[j] = w;
        ccnt[ri]--;
    }

    smat->rn = num_vertices;
    smat->cn = num_vertices;
    smat->nnz = num_edges;
    smat->indices = indices;
    smat->indptr = indptr;
    smat->data = data;

    fclose(fin);
    free(ccnt);
}

/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int Hash(char *key){
	unsigned int seed = 131;
	unsigned int hash = 0;
	while (*key) hash = hash * seed + (*key++);
	return hash % hash_table_size;
}
void InitHashTable(){
	vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;
}
void InsertHashTable(char *key, int value){
	int addr = Hash(key);
	while (vertex_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
	vertex_hash_table[addr] = value;
}
int SearchHashTable(char *key){
	int addr = Hash(key);
	while (1){
		if (vertex_hash_table[addr] == -1) return -1;
        if(!strcmp(key, vertex_id2name[vertex_hash_table[addr]])) return vertex_hash_table[addr];
		addr = (addr + 1) % hash_table_size;
	}
	return -1;
}

/* Add a vertex to the vertex set */
int AddVertex(char *name){

    int l = strlen(name) + 1;
    l = l > MAX_STRING ? MAX_STRING : l;

    string str_buf = (string)calloc(l, sizeof(char));
    strncpy(str_buf, name, l-1);
    vertex_id2name[num_vertices] = str_buf;

	vertex_outdeg[num_vertices] = 0;
	vertex_indeg[num_vertices] = 0;
	vertex_weight[num_vertices] = 0;

	num_vertices++;

	if (num_vertices + 2 >= max_num_vertices){
		max_num_vertices += 1000;
        vertex_outdeg = (int *)realloc(vertex_outdeg, max_num_vertices * sizeof(int));
        vertex_indeg = (int *)realloc(vertex_indeg, max_num_vertices * sizeof(int));
        vertex_weight = (float32 *)realloc(vertex_weight, max_num_vertices * sizeof(float32));
        vertex_id2name = (string *)realloc(vertex_id2name, max_num_vertices * sizeof(string));
	}

	InsertHashTable(name, num_vertices - 1);
	return num_vertices - 1;
}

int BiSample(struct SparseMatrix *smat, struct AliasMat *alsmat, int ri){

    if(ri < 0 || ri >= smat->rn) 
        return NULL_VERTEX_ID;

    int s = smat->indptr[ri], t = smat->indptr[ri+1];

    if(t == s) 
        return NULL_VERTEX_ID;
    if(t == s+1) 
        return smat->indices[s];

    float tmp = gsl_rng_uniform(gsl_r);
    int k = (int) (t-s)*tmp + s;
    float32 p = gsl_rng_uniform(gsl_r);
    return p < alsmat->prob[k] ? smat->indices[k] : alsmat->alias[k];
}

struct AliasMat *AliasMethod(SparseMatrix *smat){

    int rn = smat->rn, cn = smat->cn, nnz = smat->nnz;
    int * indptr = smat->indptr, *indices = smat->indices;
    int *alias = (int *)calloc(nnz, sizeof(int));
    int *overfull = (int *)malloc(cn * sizeof(int));
    int *underfull = (int *)malloc(cn * sizeof(int));
    float32 *prob = (float32 *)calloc(nnz, sizeof(float32));
    float32 *norm_prob = (float32 *)calloc(nnz, sizeof(float32));
    float32 *data = smat->data;

    if(prob == NULL || norm_prob == NULL || overfull == NULL || underfull == NULL){
    	printf("Error: memory allocation failed!\n");
		exit(1);
    } 

    float32 sum, nprob_;
    int num_underfull, num_overfull, cur_underfull, cur_overfull;

    for(int i=1,s=0,t=0,d=0; i <= rn; i++){

        s = indptr[i-1];
        t = indptr[i];
        d = t - s;

        if(d <= 1) continue;
        if(d == 1){prob[s] = 1; continue;}

        sum = 0;
        num_underfull = 0; 
        num_overfull = 0;
        cur_underfull = 0; 
        cur_overfull = 0;

        for(int j=s; j < t; j++) sum += data[j];
        for(int j=s; j < t; j++) norm_prob[j] = data[j] * d / sum;
        for(int j=s; j < t; j++){
            nprob_ = norm_prob[j];
            if(nprob_ == 1) prob[j] = 1;
            else if(nprob_ < 1) underfull[num_underfull++] = j; //{indices[j];
                //printf("- %f %d\n",norm_prob[j],indices[j]);}
            else overfull[num_overfull++] = j;//{indices[j];
                //printf("+ %f %d\n",norm_prob[j],indices[j]);}
        }
        while(num_overfull>0 && num_underfull>0){
            cur_underfull = underfull[--num_underfull];
            cur_overfull = overfull[--num_overfull];
            prob[cur_underfull] = norm_prob[cur_underfull];
            alias[cur_underfull] = indices[cur_overfull]; // index transformation
            nprob_ = norm_prob[cur_overfull];
            nprob_ = nprob_ + norm_prob[cur_underfull] - 1;
            norm_prob[cur_overfull] = nprob_;

            if(nprob_ < 1) underfull[num_underfull++] = cur_overfull;
            else if(nprob_ > 1) num_overfull++;
            else prob[cur_overfull] = 1;
        }
        // sovle the residual (precision of floating) problem e.g. 1.0001, 0.99996
        while(num_overfull) prob[overfull[--num_overfull]] = 1;
        while(num_underfull) prob[underfull[--num_underfull]]=1;
    } 

    struct AliasMat *alsmat = (struct AliasMat *)calloc(1, sizeof(struct AliasMat *));
    alsmat->prob = prob;
    alsmat->alias = alias;

    free(norm_prob);
    free(overfull);
    free(underfull);

    return alsmat;
}


void PrintSpaMat(struct SparseMatrix *smat, int s2d){

    if(!s2d){
        printf("Number of rows: %d, Number of collumns: %d, Number of elements: %d\n", 
                smat->rn, smat->cn, smat->nnz);
        return;
    }

    int rn = smat->rn, cn = smat->cn, nnz = smat->nnz;
    float32 *row = (float32 *)calloc(cn, sizeof(float32)); 
    int *indptr = smat->indptr, *indices = smat->indices;
    float32 *data = smat->data;

    for(int i=1,s=0,t=0; i <= rn; i++){
        s = indptr[i-1];
        t = indptr[i];
        for(int j=s; j < t; j++)row[indices[j]] = data[j];
        for(int j=0; j < cn; j++) printf("%f ", row[j]);
        printf("\n");
        for(int j=s; j < t; j++)row[indices[j]] = 0;
    }
    free(row);
}

/* Initialize the vertex embedding and the context embedding */
void InitVector(){

	long long a, b;

	a = posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(float32));
	if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    for(b = 0; b < num_vertices * dim; b++) emb_vertex[b] = (gsl_rng_uniform(gsl_r) - 0.5) / dim * 0.01;
    //for(b = 0; b < num_vertices * dim; b++) emb_vertex[b] = (rand() / (float32)RAND_MAX - 0.5) / dim;

    /*DEL
	a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(float32));
	if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_context[a * dim + b] = 0;
    */
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable(){
	float32 sum = 0, cur_sum = 0, por = 0, w;
	int vid = 0;
	neg_table = (int *)malloc(neg_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++){
        w = vertex_weight[k];
        if(w == 0) continue;
        sum += pow(w, NEG_SAMPLING_POWER);
    }
	for (int k = 0; k != neg_table_size;){
		if ((float32)(k + 1) / neg_table_size > por){
            w = vertex_weight[vid++];
            if(w == 0) continue;
			cur_sum += pow(w, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
		}
		neg_table[k++] = vid - 1;
	}
}

/* Fastly compute sigmoid function */
void InitSigmoidTable(){
	float32 x;
	sigmoid_table = (float32 *)malloc((sigmoid_table_size + 1) * sizeof(float32));
	for (int k = 0; k != sigmoid_table_size; k++){
		x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
		sigmoid_table[k] = 1 / (1 + exp(-x));
	}
}

float32 FastSigmoid(float32 x){
    if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed){
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings*/
void Update(int vs_idx, int target_idx, float32 *vec_error, int label){
    
    float32 f=0, f_, g, gg;
    for(int i=0; i < dim; i++){f_ = emb_vertex[vs_idx+i] - emb_vertex[target_idx+i]; f += f_ * f_;}///c2v
    if(label == 0) g = 2 * gam / (1 + f) / (0.1 + f);
    else g = -2 / (1 + f);
    for(int i=0; i < dim; i++){
        gg = g * (emb_vertex[vs_idx+i] - emb_vertex[target_idx+i]);///c2v
        if(gg > gclip) gg = gclip;
        if(gg < -gclip) gg = -gclip;
        vec_error[i] += gg * rho;
        emb_vertex[target_idx+i] -= gg * rho;///c2v
    } 
}

void *TrainThread(void *id){
    
    long long count = 0, last_count = 0;
    long long vs, vt, vs_idx, target, target_idx;
    unsigned long long seed = (long long)id;
    float32 *vec_error = (float32 *)malloc(dim * sizeof(float32));

    ////bool *sampledrcd = (bool *)calloc(num_vertices, sizeof(bool));
    while(1){

        if(count > total_samples / num_threads + 2) break;

        if (count - last_count>10000){
			current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  Progress: %.3f%%", 13, rho, 
                   (float32)current_sample_count/(float32)(total_samples + 1) * 100);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (float32)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

        // Sample a pair of vertices (vs, vt)
        vs = neg_table[Rand(seed)]; //vs = neg_table[(int)floor(gsl_rng_uniform(gsl_r) * neg_table_size)];
        vt = BiSample(smat, alsmat, vs);
        if(vt == NULL_VERTEX_ID || vt == vs) continue;
        //// sampledrcd[vs] = true;
        
        // Negtive Sampling
        vs_idx = vs * dim;
        for(int i = 0; i < dim; i++) vec_error[i] = 0;
        target = vt;
        for(int i = 0, label = 1; i < num_negative + 1;){
            if(i > 0){
                target = neg_table[Rand(seed)]; //target = neg_table[(int)floor(gsl_rng_uniform(gsl_r) * neg_table_size)];
                if(vt == target || vs == target) continue;
                label = 0;
            } 
            target_idx = target * dim;
            Update(vs_idx, target_idx, vec_error, label);
            i++;
        }
        for(int i = 0; i < dim; i++) emb_vertex[vs_idx+i] += vec_error[i];
        count++;
    }
    /*int cnt = 0;
    for(int i=0; i < num_vertices; i++) if(sampledrcd[i]) cnt++;
    printf("count %d\n", cnt);*/

    free(vec_error);
	pthread_exit(NULL);
}
void TrainModel(){
    long ti;

    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    printf("--------------------------------\n");
	printf("Samples: %lldM\n", total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("--------------------------------\n");

    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, (unsigned long)(time(NULL)));

    InitHashTable();
    ReadData();        
    InitAlias();
    InitVector();
	InitNegTable();
	InitSigmoidTable();

	clock_t start = clock();
    printf("--------------------------------\n");    
	for (ti = 0; ti < num_threads; ti++) pthread_create(&pt[ti], NULL, TrainThread, (void *)ti);
	for (ti = 0; ti < num_threads; ti++) pthread_join(pt[ti], NULL);
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

    Output();
}
void Output(){
	FILE *fo = fopen(embedding_file, "wb");
    int embcnt = 0;
    for(int a=0; a < num_vertices; a++){if(vertex_outdeg[a]==0) continue; embcnt++;}/// OPTION
	fprintf(fo, "%d %d\n", embcnt, dim);
	for (int a = 0; a < num_vertices; a++){
        if(vertex_outdeg[a] == 0) continue;/// OPTION
		fprintf(fo, "%s ", vertex_id2name[a]);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(float32), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%f ", emb_vertex[a * dim + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}
int ArgPos(char *str, int argc, char **argv){
	int a;
	for(a = 1; a < argc; a++){
        if(!strcmp(str, argv[a])){
            if(a == argc - 1){printf("Argument missing for %s\n", str); exit(1);}
            return a;
        }
	}
	return -1;
}


int main(int argc, char **argv) {
    int i;
    char mpath[MAX_STRING];

    if (argc == 1) {
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <directory>\n");
		printf("\t\tUse network data from <directory> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the learnt embeddings\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
		printf("\t-size <int>\n");
		printf("\t\tSet dimension of vertex embeddings; default is 2\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5\n");
		printf("\t-samples <int>\n");
		printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-rho <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\t-gamma <float>\n");
		printf("\t\tSet the gamma; default is 7\n");
		printf("\nExamples:\n");
		printf("./gsne_hom -train data_directory -output emb.txt -binary 1 -size 2 -negative 5 -samples 10 -rho 0.025 -threads 10\n\n");
		return 0;
	}

    if ((i = ArgPos((char *)"-train", argc, argv)) > 0){strcpy(graph_file, argv[i + 1]);}
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]) == 0 ? false : true;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-gamma", argc, argv)) > 0) gam = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

    total_samples *= 1000000;
    rho = init_rho;

    /* Testing Part
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, (unsigned long)(time(NULL)));
    for(int i=0; i < 10; i++) printf("%f\n", gsl_rng_uniform(gsl_r));*/
    
    TrainModel();
    return 0;
}

/* 
2018-01-05 problem#1: unstable performance, probably maybe caused by initialization ?
2018-01-06 solved problem#1: forget to initialize f in update function
2018-01-06 discovery#1: smaller gamma will improve the final layout*/
