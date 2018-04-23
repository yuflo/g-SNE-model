# g-SNE-model
Graph Stochastic Neighbor Embedding

### Introduction
This algorithm is partly inspired by [LargVis](https://arxiv.org/abs/1602.00370) and can be used to visualize large-scale homogeneous/heterogeneous information network.  

### Run
For homogeneous information network:
```
./gsne_hom -train data_directory -output emb.txt -binary 1 -size 2 -negative 5 -samples 10 -rho 0.025 -threads 10
```
For heterogeneous information network:
```
./gsne_hin -train data_directory -output emb.txt -binary 1 -size 2 -negative 5 -samples 10 -rho 0.025 -threads 10
```

- -train, the input file of network data;
- -output, the output file of the embedding;
- -binary, save the learnt embeddings in binary moded; default is 0 (off);
- -size, set dimension of vertex embeddings, default is 18;
- -negative, the number of negative samples used in negative sampling; the deault is 5;
- -samples, set the number of training samples as <int>Million; default is 1;
- -threads, the total number of threads used; the default is 1.
- -rho, the starting value of the learning rate; the default is 0.025;
- -gamma, set the gamma, default is 7

### Contact: 
If you have any questions about the codes and data, please feel free to contact us.
```
Chen Li, hblouis@hotmail.com
```