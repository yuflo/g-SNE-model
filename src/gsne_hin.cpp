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

struct HINProjection{
    string mpath;
    string *mpairs;
    string *bigraphs;
    int mpathlen;
    int num_bigraph;
    P2SparseMatrix *smatlis;
    P2AliasMat *alsmatlis;
    P2SparseMatrix *path_smats;
    P2AliasMat *path_alsmats;
};

//////
struct AliasMat *AliasMethod(SparseMatrix *smat);
int BiSample(struct SparseMatrix *smat, struct AliasMat *alsmat, int ri);
struct SparseMatrix *LoadSpaMat(string csrmat_file);
void PrintSpaMat(struct SparseMatrix *smat, int s2d);
void Update(int vs_idx, int vt_idx, float32 *vec_error, int label);
float32 FastSigmoid(float32 x);
int Rand(unsigned long long &seed);
string *SeparatePath(string path, int &num_pairs);
void ReadData();
bool InitHinproj(string mpath);
void InitPathAlias();
void *TrainThread(void *id);
void TrainModel(string mpath);
int ArgPos(char *str, int argc, char **argv);
void Output();

//////
const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

int num_threads = 10, num_negative = 5, dim = 2;
int num_vertices;
int *neg_table;
long long total_samples = 1, current_sample_count = 0;
float32 init_rho = 1, rho, gam= 7, gclip=5;
float32 *emb_vertex, *emb_context, *sigmoid_table;
float32 *vertex_path_degree;
char ddir[MAX_STRING];
char embedding_file[MAX_STRING];
string *vertex_id2name;
struct HINProjection *hinproj;
bool is_binary = false;

//////

/*
void sampletest(int vi){
    unsigned long long seed = (long long)vi;
    int vs = vi;
    int vt = vs;
    int num_mpairs = hinproj->mpathlen-1;
    P2SparseMatrix *path_smats = hinproj->path_smats;
    P2AliasMat *path_alsmats = hinproj->path_alsmats;

    for(int k = 0; k < num_mpairs && vt != NULL_VERTEX_ID; k++){
        vt = BiSample(path_smats[k], path_alsmats[k], vt);
        printf("%d\n", vt);
    }
}*/

// 3 types of num_pairs value: zero -> no pair, positive -> normal, negative (-1) -> asymmetric path
string *SeparatePath(string path, int &num_pairs){

    int i = 0, k = 0, max_num_pairs = 3;
    int *split_idx = (int *)calloc(strlen(path), sizeof(int));
    num_pairs = 0;

    while(path[i] != '\0'){
        if(path[i] == '-') split_idx[++k] = i + 1;
        i++;
    }
    split_idx[++k] = i + 1;

    if(k < 2) return NULL;
    if(k % 2 == 0){num_pairs--; return NULL;} // asymmetric

    string *vertex_pairs = (string *)calloc(max_num_pairs, sizeof(string));

    // symmetric check
    for(int j1=0, j2=k-1, s1=0, s2=0, t1=0, t2=0; j1 != j2; j1++, j2--){
        s1 = split_idx[j1];
        s2 = split_idx[j2];
        t1 = split_idx[j1+1] - 2;
        t2 = split_idx[j2+1] - 2;
        if(t1 - s1 != t2 - s2){num_pairs--; return NULL;}
        for(int d=0; d <= t1 - s1; d++) if(path[s1+d] != path[s2+d]){num_pairs--; return NULL;}
    }

    for(int j=1, s=0, t=0; j < k; j++){
        s = split_idx[j-1];
        t = split_idx[j+1] - 2;
        if(++num_pairs > max_num_pairs){
            max_num_pairs += 3;
            vertex_pairs = (string *)realloc(vertex_pairs, max_num_pairs * sizeof(string));
        }
        char *strtmp = (char *)calloc(t-s+2, sizeof(char));
        strncpy(strtmp, path+s, t-s+1);
        vertex_pairs[num_pairs-1] = strtmp;
    }

    return vertex_pairs;
}

bool InitHinproj(string mpath){

    if(strlen(mpath) == 0){printf("Invalid meta path\n"); return false;}

    int num_pairs;
    string *mpairs = SeparatePath(mpath, num_pairs); 
    if(num_pairs == 0){
        printf("The length of given meta path should be larger than 3.\n");
        return false;
    }
    if(num_pairs == -1){
        printf("The given meta path %s is asymmetric.\n", mpath);
        return false;
    }

    FILE *fin;
    char str[100];
    int num_checked = num_pairs;
    bool checked[num_pairs];
    string filename = (string)calloc(100, sizeof(char));
    char mpath_file[] = "meta-path.txt";
    strcpy(filename, ddir);
    strcat(filename, mpath_file);
    fin = fopen(filename, "rb");
    if (fin == NULL){printf("ERROR: meta-path file not found!\n"); exit(1);}
    for(int k=0; k < num_pairs; k++) checked[k] = false;
    while(fscanf(fin, "%s", str) != EOF){
        for(int k=0; k < num_pairs; k++)
            if(strcmp(str, mpairs[k]) == 0){num_checked--; checked[k]=true; break;}
        if(num_checked == 0) break;
    }
    fclose(fin);
    if(num_checked > 0){
        printf("non-existent meta pair: ");
        for(int k=0; k < num_pairs; k++) if(!checked[k]){printf("%s ", mpairs[k]);}
        printf("\n");
        return false;
    }

    string *bigraphs = (string *)calloc(num_pairs, sizeof(string));
    int i=0, j=0, k=0;
    for(; j < num_pairs; j++){
        for(k=0; k < i; k++) if(strcmp(bigraphs[k], mpairs[j]) == 0) break;
        if(k == i){
            bigraphs[i] = (string)calloc(strlen(mpairs[j])+1, sizeof(char));
            strcpy(bigraphs[i], mpairs[j]);
            i++;
        }
    }

    hinproj = (struct HINProjection *)malloc(sizeof(struct HINProjection));
    hinproj->mpath = mpath;
    hinproj->mpathlen = num_pairs + 1;
    hinproj->num_bigraph = i;
    hinproj->mpairs = mpairs;
    hinproj->bigraphs = bigraphs;
    hinproj->smatlis = (P2SparseMatrix *)calloc(i, sizeof(P2SparseMatrix));
    hinproj->alsmatlis = (P2AliasMat *)calloc(i, sizeof(P2AliasMat));
    hinproj->path_alsmats = (P2AliasMat *)calloc(num_pairs, sizeof(P2AliasMat));
    hinproj->path_smats = (P2SparseMatrix *)calloc(num_pairs, sizeof(P2SparseMatrix));

    return true;
}

void ReadData(){
    FILE *fin;
    string filename = (string)calloc(100, sizeof(char));
    string vname = (string)calloc(100, sizeof(char));
    string *bigraphs = hinproj->bigraphs;
    char extension_csrm[] = ".csrm";
    char extension_map[] = ".map";
    P2SparseMatrix *smatlis = hinproj->smatlis;
    int vid;

    for(int k = 0; k < hinproj->mpathlen-1; k++){
        strcpy(filename, ddir);
        strcat(filename, bigraphs[k]);
        strcat(filename, extension_csrm);
        fin = fopen(filename, "rb");
        if (fin == NULL){
            printf("ERROR: file %s not found!\n", filename);
            exit(1);
        }
        smatlis[k] = LoadSpaMat(filename);
        fclose(fin);

        if(k == 0){
            int c = 0;
            num_vertices = smatlis[k]->rn;
            vertex_id2name = (string *)malloc(num_vertices * sizeof(string *)); 
            while(bigraphs[k][c] != '-') c++;
            strcpy(filename, ddir);
            strncat(filename, bigraphs[k], c);
            strcat(filename, extension_map);
            fin = fopen(filename, "rb");
            if (fin == NULL){
                printf("ERROR: file %s not found!\n", filename);
                exit(1);
            }
            for(int i=0; i < num_vertices; i++){
                fscanf(fin, "%d %s", &vid, vname);
                vertex_id2name[vid] = (string)calloc(strlen(vname)+1, sizeof(char));
                strcpy(vertex_id2name[vid], vname);
            }
            fclose(fin);

            int *indices = smatlis[k]->indices, *indptr = smatlis[k]->indptr;
            float32 *data = smatlis[k]->data;
            vertex_path_degree = (float32 *)malloc(num_vertices * sizeof(float32));
            float32 w = 0;
            for(int i=0,s=0,t=0; i < num_vertices; i++){
                s = indptr[i];
                t = indptr[i+1];
                for(int j=s; j < t; j++) w += data[j];
                vertex_path_degree[i] = w;
                w = 0;
            }
        }
    }
}
void InitPathAlias(){
    for(int k=0; k < hinproj->num_bigraph; k++)
        hinproj->alsmatlis[k] = AliasMethod(hinproj->smatlis[k]);
    for(int k=0, j=0; k < hinproj->mpathlen-1; k++){
        for(j=0; strcmp(hinproj->mpairs[k], hinproj->bigraphs[j]) != 0; j++);
        hinproj->path_alsmats[k] = hinproj->alsmatlis[j]; 
        hinproj->path_smats[k] = hinproj->smatlis[j]; 
    }
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

struct SparseMatrix *LoadSpaMat(string csrmat_file){

    FILE *fin;
    int rn, cn, nnz;
    int *indices, *indptr;
    float32 *data;
    struct SparseMatrix *smat;

    fin = fopen(csrmat_file, "rb");
    if (fin == NULL)
	{
		printf("ERROR: csr matrix file not found!\n");
		exit(1);
	}

    fscanf(fin, "%d %d %d\n", &rn, &cn, &nnz);
    printf("Number of rows: %d, Number of collumns: %d, Number of elements: %d\n", rn, cn, nnz);

    indptr = (int *)malloc((rn+1) * sizeof(int));
    indices = (int *)malloc(nnz * sizeof(int));
    data = (float32 *)malloc(nnz * sizeof(float32));

    if(indices == NULL || indptr == NULL || data == NULL){
		printf("Error: memory allocation failed!\n");
        exit(1);
    }

    for(int i=0; i <= rn; i++)fscanf(fin, "%d", indptr+i);
    for(int i=0; i < nnz; i++)fscanf(fin, "%d", indices+i);
    for(int i=0; i < nnz; i++)fscanf(fin, "%f", data+i);

    smat = (struct SparseMatrix *)calloc(1, sizeof(struct SparseMatrix));
    smat->rn = rn;
    smat->cn = cn;
    smat->nnz = nnz;
    smat->indices = indices;
    smat->indptr = indptr;
    smat->data = data;

    fclose(fin);

    return smat;
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
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
        emb_vertex[a * dim + b] = (rand() / (float32)RAND_MAX - 0.5) / dim;

    /*///
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
        w = vertex_path_degree[k];
        if(w == 0) continue;
        sum += pow(w, NEG_SAMPLING_POWER);
    }
	for (int k = 0; k != neg_table_size;){
		if ((float32)(k + 1) / neg_table_size > por){
            w = vertex_path_degree[vid++];
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

        gg = g * (emb_vertex[target_idx+i] - emb_vertex[vs_idx+i]);///c2v
        if(gg > gclip) gg = gclip;
        if(gg < -gclip) gg = -gclip;
        emb_vertex[target_idx+i] += gg * rho;///c2v
    } 
}

void *TrainThread(void *id){
    
    int num_mpairs = hinproj->mpathlen-1;
    long long count = 0, last_count = 0;
    long long vs, vt, vs_idx, target, target_idx;
    unsigned long long seed = (long long)id;
    float32 *vec_error = (float32 *)malloc(dim * sizeof(float32));
    bool *sampled = (bool*)calloc(num_vertices, sizeof(bool));
    bool sample_state;
    P2SparseMatrix *path_smats = hinproj->path_smats;
    P2AliasMat *path_alsmats = hinproj->path_alsmats;

    //// bool *sampledrcd = (bool *)calloc(num_vertices, sizeof(bool));
    while(1){

        if(count > total_samples / num_threads + 2) break;

        if (count - last_count>10000){
			current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  Progress: %.3lf%%", 13, rho, 
                   (float32)current_sample_count/(float32)(total_samples + 1) * 100);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (float32)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

        // sample a pair of vertices (vs, vt)
        vs = neg_table[Rand(seed)];
        vt = vs;
        for(int k = 0; k < num_mpairs && vt != NULL_VERTEX_ID; k++) vt = BiSample(path_smats[k], path_alsmats[k], vt);
        if(vt == NULL_VERTEX_ID || vt == vs) continue;
        //// sampledrcd[vs] = true;
        
        // Negtive Sampling
        vs_idx = vs * dim;
        for(int i = 0; i < dim; i++) vec_error[i] = 0;
        target = vt;
        for(int i = 0, label=1; i < num_negative;){
            if(i > 0){
                target = neg_table[Rand(seed)];
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
void TrainModel(string mpath){
    long ti;

    bool state1 = InitHinproj(mpath);
    if(!state1) exit(1);

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
    gsl_rng_set(gsl_r, 314159265);

    ReadData();        
    InitPathAlias();
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
    for(int a=0; a < num_vertices; a++){if(vertex_path_degree[a]==0) continue; embcnt++;}
	fprintf(fo, "%d %d\n", embcnt, dim);
	for (int a = 0; a < num_vertices; a++){
        if(vertex_path_degree[a] == 0) continue;
		fprintf(fo, "%s ", vertex_id2name[a]);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(float32), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
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
		printf("./gsne_hin -train data_directory -output emb.txt -binary 1 -size 2 -negative 5 -samples 10 -rho 0.025 -threads 10\n\n");
		return 0;
	}

    if ((i = ArgPos((char *)"-train", argc, argv)) > 0){strcpy(ddir, argv[i + 1]); strcat(ddir,"/");}
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
    gsl_rng_set(gsl_r, 314159265);
    struct SparseMatrix *smat;
    struct AliasMat *alsmat;

    smat = LoadSpaMat(argv[1]);
    printf("%d %d %d\n", smat->rn, smat->cn, smat->nnz);
    PrintSpaMat(smat, true);
    alsmat = AliasMethod(smat);
    for(int j=0; j <= smat->rn; j++) printf("%d ", smat->indptr[j]);
    printf("\n");
    for(int j=0; j < smat->nnz; j++) printf("%f ", alsmat->prob[j]);
    printf("\n");
    int ri;
    while(scanf("%d", &ri) != EOF)
        printf("%d\n", BiSample(hinproj->path_smats[0], hinproj->path_alsmats[0], ri));
    for(int i=0; i < hinproj->mpathlen-1; i++) printf("%s\n", hinproj->mpairs[i]);
    for(int i=0; i < hinproj->num_bigraph; i++) printf("%s\n", hinproj->bigraphs[i]);
    */
    printf("--------------------------------\nPlease input an appropriate symmetric meta-path:\n");
    scanf("%s", mpath);

    /* Testing Part
    bool tmp = InitHinproj(mpath);
    if(!tmp) exit(1);
    ReadData();
    InitPathAlias();
    smat = hinproj->path_smats[1];
    alsmat = hinproj->path_alsmats[1];
    for(int j=0; j <= smat->rn; j++) printf("%d ", smat->indptr[j]);
    printf("\n");
    for(int j=0; j < smat->nnz; j++) printf("%f ", alsmat->prob[j]);
    printf("\n");
    */

    TrainModel(mpath);

    return 0;
}
