#include "printFunctions.cu"
#include <time.h>

#define THREADS_PER_BLOCK 1024

char * copySequencesToDevice (char ** sequences, int numSequences, int sequenceLength) {
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

__global__ void createBucketSequence (char * bucketSequence, char * d_sequences, int sequenceLength, int bucketSequenceIndex) {
  for (int i = threadIdx.x; i < sequenceLength; i += blockDim.x)
    *(bucketSequence + i) = *(d_sequences + bucketSequenceIndex * sequenceLength + i);
}

__global__ void assignBuckets (char * sequences, char * bucketSequence, uint * bucketCounts, int numSequences, int sequenceLength, int numBuckets, int matchLength, double matchAccuracy, int bucketsPerThread, int bucketGroup) {

  int numMatches = 0;
  int bucketIndex = threadIdx.x + bucketGroup * numBuckets / (float) bucketsPerThread;
	    
  // use shared memory for the sequence and buckets
  extern __shared__ char shared[];
  char * sharedSequence = &shared[0];
  char * sharedBucketSequence = &shared[sequenceLength];
  
  // fill sharedSequence and sharedBucketSequence
  for (int i = threadIdx.x; i < sequenceLength; i += blockDim.x) {
    sharedSequence[i] = sequences[blockIdx.x * sequenceLength + i];
    sharedBucketSequence[i] = *(bucketSequence + i);
  }
  
  syncthreads();

  if (bucketIndex < numBuckets) 
    for (int i = 0; i < numBuckets; i++) {
      for (int j = 0; j < matchLength; j++)
        if (*(sharedBucketSequence + bucketIndex  + j) == *(sharedSequence + i + j))
          numMatches++;

      if (numMatches / (double) matchLength >= matchAccuracy) {
        atomicInc (bucketCounts + bucketIndex, UINT_MAX);
        return;
      }
      //  atomicAdd (bucketCounts + bucketIndex, (numMatches / (double) matchLength >= matchAccuracy)); 
      numMatches = 0;
    }  
}

uint * sequencer (char * d_sequences, int numSequences, int sequenceLength, int matchLength, double matchAccuracy) {

  // printSequences (sequences, numSequences, sequenceLength);
  // printDeviceSequences (d_sequences, numSequences, sequenceLength);

  // choose a random sequence to create buckets from
  srand (time (NULL));
  int bucketSequence = 0;//rand() % numSequences;
	  
  // printf ("bucketSequence = %d\n", bucketSequence);

  // create the buckets
  int numBuckets = sequenceLength - matchLength + 1;
  char * d_bucketSequence;
  cudaMalloc (&d_bucketSequence, sizeof (char) * sequenceLength);
  int numThreads = THREADS_PER_BLOCK;
  //  if (numThreads > sequenceLength)
  //   numThreads = sequenceLength;

  createBucketSequence<<<ceil (sequenceLength / (float) numThreads), numThreads>>> (d_bucketSequence, d_sequences, sequenceLength, bucketSequence);

  // printDeviceSequences (d_bucketSequence, 1, sequenceLength);

  // numThreads = THREADS_PER_BLOCK;
  int bucketsPerThread = ceil (numBuckets / (float) numThreads);
  // if (numThreads > numBuckets)
  //  numThreads = numBuckets;

  // make counters for each bucket, with the last one counting how many didn't fit
  // into any buckets
  uint * d_bucketCounts;
  cudaMalloc (&d_bucketCounts, sizeof (uint) * numBuckets * bucketsPerThread);
  cudaMemset (d_bucketCounts, 0, sizeof (uint) * numBuckets * bucketsPerThread);

  
  printf ("numThreads = %d, numBlocks = %d, numShared = %d, numBuckets = %d, bucketsPerThread = %d\n", numThreads, numSequences, sequenceLength * 2, numBuckets, bucketsPerThread);

  printFirstLastBuckets (d_bucketSequence, numBuckets, matchLength, sequenceLength);
  printDeviceFirstLast (d_sequences, numSequences, sequenceLength);
  
  // each block is a sequence
  // each thread assigns bucketsPerThread number of buckets
  for (int i = 0; i < bucketsPerThread; i++) {
    assignBuckets<<<numSequences, numThreads, sizeof (char) * sequenceLength * 2>>> (d_sequences, d_bucketSequence, d_bucketCounts, numSequences, sequenceLength, numBuckets, matchLength, matchAccuracy, bucketsPerThread, i);
    cudaThreadSynchronize();
  }

  
  printf("\nnow printing after assignBuckets:\n\n");
  printFirstLastBuckets (d_bucketSequence, numBuckets, matchLength, sequenceLength); 
  printDeviceFirstLast (d_sequences, numSequences, sequenceLength);
  
  uint * bucketCounts = (uint *) malloc (sizeof (uint) * numBuckets);
  cudaMemcpy (bucketCounts, d_bucketCounts, sizeof (uint) * numBuckets, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < numBuckets + 1; i++)
  // printf ("bucketCounts[%d] = %u\n", i, *(bucketCounts + i));


  // printDeviceSequences (d_buckets, numBuckets, matchLength);

  // run kernel in loop from length of sequence down to ~10 or so to see
  // which bucket sizes give good results
  // will need an array which holds what the matching pattern is
  // will need an array to store data of which sequences have matching pattern
	  
  cudaFree (d_bucketCounts);
  cudaFree (d_bucketSequence);

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
  char * d_sequences = copySequencesToDevice (sequences, numSequences, sequenceLength);

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
