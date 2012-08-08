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

__global__ void createBuckets (char * sequence, char * buckets, int sequenceLength, int numBuckets, int matchLength, int bucketsPerThread) {
	  
  int index = threadIdx.x + (blockIdx.x % bucketsPerThread * blockDim.x);
 
  if (index < numBuckets)
    for (int i = 0; i < matchLength; i++)
      *(buckets + matchLength * index + i) = *(sequence + index + i);
}

__global__ void assignBuckets (char * sequences, char * buckets, uint * bucketCounts, int numSequences, int sequenceLength, int numBuckets, int matchLength, double matchAccuracy, int bucketsPerThread) {
	  
	  
  // read buckets into shared memory for faster access
  extern __shared__ char sharedSequence[];
  for (int i = threadIdx.x; i < sequenceLength; i += blockDim.x)
    sharedSequence[i] = sequences[blockIdx.x * sequenceLength + i];
  
  // if (threadIdx.x == 0 && blockIdx.x == 1)
  //  printf ("shared sequence == %s\n", sharedSequence);

  syncthreads();
  
  int numMatches = 0;
  int bucketIndex;

  for (int k = 1; k <= bucketsPerThread; k++) {
    for (int i = 0; i < numBuckets; i++) {
      for (int j = 0; j < matchLength; j++) {
        if ((bucketIndex = threadIdx.x * k * matchLength + j) < numBuckets * matchLength) {
          if (*(buckets + bucketIndex) == *(sharedSequence + i + j))
            numMatches++;

          if (numMatches / (double) matchLength >= matchAccuracy) {
            atomicInc (bucketCounts + threadIdx.x * k, UINT_MAX);
            // return;
          }

        }
      }  

      numMatches = 0;
    }
  }
  
  // atomicInc (bucketCounts + numBuckets, UINT_MAX);
}

uint * sequencer (char * d_sequences, int numSequences, int sequenceLength, int matchLength, double matchAccuracy) {

  // printSequences (sequences, numSequences, sequenceLength);
  // printDeviceSequences (d_sequences, numSequences, sequenceLength);

  // choose a random sequence to create buckets from
  srand (time (NULL));
  int bucketSequence = 0;//rand() % numSequences;
	  
  // printf ("bucketSequence = %d\n", bucketSequence);

  // create the buckets
  char * d_buckets;
  int numBuckets = sequenceLength - matchLength + 1;
  cudaMalloc (&d_buckets, sizeof (char) * numBuckets * matchLength);

  int numThreads = THREADS_PER_BLOCK;
  int bucketsPerThread = ceil (numBuckets / (float) numThreads);

  if (numThreads > numBuckets)
    numThreads = numBuckets;

  createBuckets<<<bucketsPerThread, numThreads>>> (d_sequences + bucketSequence * sequenceLength, d_buckets, numBuckets, sequenceLength, matchLength, bucketsPerThread);

  printDeviceFirstLast (d_sequences, numSequences, sequenceLength);
  printDeviceFirstLast (d_buckets, numBuckets, matchLength);

  // make counters for each bucket, with the last one counting how many didn't fit
  // into any buckets
  uint * d_bucketCounts;
  cudaMalloc (&d_bucketCounts, sizeof (uint) * (numBuckets + 1));
  cudaMemset (d_bucketCounts, 0, sizeof (uint) * (numBuckets + 1));

  // each block is a sequence
  // each thread assigns bucketsPerThread number of buckets
  assignBuckets<<<numSequences, numThreads, sizeof (char) * sequenceLength>>> (d_sequences, d_buckets, d_bucketCounts, numSequences, sequenceLength, numBuckets, matchLength, matchAccuracy, bucketsPerThread);

  cudaThreadSynchronize();

  uint * bucketCounts = (uint *) malloc (sizeof (uint) * (numBuckets + 1));
  cudaMemcpy (bucketCounts, d_bucketCounts, sizeof (uint) * (numBuckets + 1), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < numBuckets + 1; i++)
  // printf ("bucketCounts[%d] = %u\n", i, *(bucketCounts + i));
  printf("\nnow printing after assignBuckets:\n");
  printDeviceFirstLast (d_sequences, numSequences, sequenceLength);
  printDeviceFirstLast (d_buckets, numBuckets, matchLength); 

  // printDeviceSequences (d_buckets, numBuckets, matchLength);

  // run kernel in loop from length of sequence down to ~10 or so to see
  // which bucket sizes give good results
  // will need an array which holds what the matching pattern is
  // will need an array to store data of which sequences have matching pattern
	  
  cudaFree (d_bucketCounts);
  cudaFree (d_buckets);

  return bucketCounts;
}

__global__ void counterKernel (char * sequences, int sequenceLength, char * query, int queryLength, uint * count, double matchAccuracy) {

  // read query into shared memory for faster access
  extern __shared__ char sharedQuery[];
  if (threadIdx.x < queryLength)
    *(sharedQuery + threadIdx.x) = query[threadIdx.x];

  int numMatches = 0;
  int startSpot = sequenceLength * blockIdx.x + threadIdx.x;

  for (int i = 0; i < queryLength; i++) {
    if (*(sequences + startSpot + i) == *(query + i))
      numMatches++;
  }

  if (numMatches / (double) queryLength >= matchAccuracy)
    atomicInc (count, UINT_MAX);
}


uint counter (char ** sequences, int numSequences, int sequenceLength, char * query, int queryLength, double matchAccuracy) {
	 
  // put sequences into device memory
  char * d_sequences = copySequenceToDevice (sequences, numSequences, sequenceLength);

  // put query into device memory
  char * d_query;
  cudaMalloc (&d_query, queryLength * sizeof (char));
  cudaMemcpy (d_query, query, queryLength * sizeof (char), cudaMemcpyHostToDevice);

  // counts of how many times the query was found
  uint count = 0;
  uint * d_count;
  cudaMalloc (&d_count, sizeof (uint));
  cudaMemcpy (d_count, &count, sizeof (uint), cudaMemcpyHostToDevice);

  counterKernel<<<numSequences, sequenceLength - queryLength + 1, queryLength * sizeof (char)>>> (d_sequences, sequenceLength, d_query, queryLength, d_count, matchAccuracy);

  cudaMemcpy (&count, d_count, sizeof (uint), cudaMemcpyDeviceToHost);

  cudaFree (d_count);
  cudaFree (d_query);
  cudaFree (d_sequences);

  return count;
}
