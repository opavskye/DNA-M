#include "printFunctions.cu"
#include <time.h>

#define THREADS_PER_BLOCK 1024

char * copySequenceToDevice (char ** sequences, int numSequences, int sequenceLength) {
  char * d_sequences;
  cudaMalloc (&d_sequences, sizeof (char) * sequenceLength * numSequences);

  for (int i = 0; i < numSequences; i++)
    cudaMemcpy (d_sequences + i * sequenceLength, *(sequences + i), sizeof (char) * sequenceLength, cudaMemcpyHostToDevice); 

  return d_sequences;
}

__global__ void createBuckets (char * sequence, char * buckets, int sequenceLength, int matchLength) {
  
  int index = threadIdx.x + blockIdx.x * threadIdx.x;

  for (int i = 0; i < matchLength; i++)
    *(buckets + matchLength * index + i) = *(sequence + index + i);
}

__global__ void assignBuckets (char * sequences, char * buckets, uint * bucketCounts, int numSequences, int sequenceLength, int numBuckets, int matchLength, double matchAccuracy) {
  
  
  // read buckets into shared memory for faster access
  extern __shared__ char sharedBuckets[];
  for (int i = threadIdx.x; i < numBuckets * matchLength; i += blockDim.x)
    if (i < numBuckets)
      for (int j = 0; j < matchLength; j++)
        sharedBuckets[i * matchLength + j] = buckets[i * matchLength + j];
  

  int numMatches = 0;

  for (int i = 0; i < numBuckets; i++) {
    for (int j = 0; j < matchLength; j++) 
      if (*(sequences + blockIdx.x * sequenceLength + threadIdx.x + j) == *(sharedBuckets + i * matchLength + j))
        numMatches++;
    
    if (numMatches / (double) matchLength >= matchAccuracy)
      atomicInc (bucketCounts + i, UINT_MAX);

    numMatches = 0;
  }    

  atomicInc (bucketCounts + numBuckets, UINT_MAX);
}

void sequencer (char ** sequences, int numSequences, int sequenceLength, int matchLength, double matchAccuracy) {

  // put sequences into device memory
  char * d_sequences = copySequenceToDevice (sequences, numSequences, sequenceLength);

  // printSequences (sequences, numSequences, sequenceLength);
  // printDeviceSequences (d_sequences, numSequences, sequenceLength);

  // choose a random sequence to create buckets from
  srand (time (NULL));
  int bucketSequence = rand() % numSequences;
  
  // printf ("bucketSequence = %d\n", bucketSequence);

  // create the buckets
  char * d_buckets;
  int numBuckets = sequenceLength - matchLength + 1;
  cudaMalloc (&d_buckets, sizeof (char) * numBuckets * matchLength); 

  int numThreads = THREADS_PER_BLOCK;
  int numBlocks = ceil (numBuckets / (float) numThreads);

  if (numThreads > numBuckets)
    numThreads = numBuckets;

  createBuckets<<<numBlocks, numThreads>>> (d_sequences + bucketSequence * sequenceLength, d_buckets, sequenceLength, matchLength);

  // make counters for each bucket, with the last one counting how many didn't fit
  //  into any buckets
  uint * d_bucketCounts;
  cudaMalloc (&d_bucketCounts, sizeof (uint) * (numBuckets + 1));
  cudaMemset (d_bucketCounts, 0, sizeof (uint) * (numBuckets + 1));

  // count how many sequences go into each bucket
  numThreads = numBuckets;
  numBlocks = numSequences;
  assignBuckets<<<numBlocks, numThreads, (sizeof (char) * numBuckets * matchLength)>>> (d_sequences, d_buckets, d_bucketCounts, numSequences, sequenceLength, numBuckets, matchLength, matchAccuracy);

  uint * bucketCounts = (uint *) malloc (sizeof (uint) * (numBuckets + 1));
  cudaMemcpy (bucketCounts, d_bucketCounts, sizeof (uint) * (numBuckets + 1), cudaMemcpyDeviceToHost);

  for (int i = 0; i < numBuckets + 1; i++)
    printf ("bucketCount[%d] = %u\n", i, *(bucketCounts + i));
  

  // printDeviceSequences (d_buckets, numBuckets, matchLength);

  // run kernel in loop from length of sequence down to ~10 or so to see
  //  which bucket sizes give good results
  //  will need an array which holds what the matching pattern is
  //  will need an array to store data of which sequences have matching pattern
  

  free (bucketCounts);
  cudaFree (d_bucketCounts);
  cudaFree (d_buckets);
  cudaFree (d_sequences);

}
