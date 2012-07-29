#include "printFunctions.cu"
#include <time.h>

#define MAX_THREADS_PER_BLOCK 1024

char * copySequenceToDevice (char ** sequences, int numSequences, int sequenceLength) {
  char * d_sequences;
  cudaMalloc (&d_sequences, sizeof (char) * sequenceLength * numSequences);

  for (int i = 0; i < numSequences; i++)
    cudaMemcpy (d_sequences + i * sequenceLength, *(sequences + i), sizeof (char) * (sequenceLength - 1), cudaMemcpyHostToDevice); 

  return d_sequences;
}

__global__ void createBuckets (char * sequence, char * buckets, int sequenceLength, int matchLength) {
  
  int index = threadIdx.x + blockIdx.x * threadIdx.x;

  // TODO: make this more coalesced later
  for (int i = 0; i < matchLength; i++)
    *(buckets + matchLength * index + i) = *(sequence + index + i);
}

void sequencer (char ** sequences, int numSequences, int sequenceLength, int matchLength, double matchAccuracy) {

  // put sequences into device memory
  char * d_sequences = copySequenceToDevice (sequences, numSequences, sequenceLength);

  // printSequences (sequences, numSequences, sequenceLength);
  // printDeviceSequences (d_sequences, numSequences, sequenceLength);


  // choose a random sequence to create buckets from
  srand (time (NULL));
  int bucketSequence = rand() % numSequences;
  
  printf("bucketSequence = %d\n", bucketSequence);

  // create the buckets
  char * d_buckets;
  int numBuckets = sequenceLength - matchLength + 1;
  cudaMalloc (&d_buckets, sizeof (char) * numBuckets * matchLength); 

  int numThreads = MAX_THREADS_PER_BLOCK;
  int numBlocks = ceil (numBuckets / (float) numThreads);

  if (numThreads > numBuckets)
    numThreads = numBuckets;

  createBuckets<<<numBlocks, numThreads>>> (d_sequences + bucketSequence * sequenceLength, d_buckets, sequenceLength, matchLength);

  printDeviceSequences (d_buckets, numBuckets, matchLength);

  // run kernel in loop from length of sequence down to ~10 or so to see
  //  which bucket sizes give good results
  //  will need an array which holds what the matching pattern is
  //  will need an array to store data of which sequences have matching pattern
  

  cudaFree (d_buckets);
  cudaFree (d_sequences);

}
