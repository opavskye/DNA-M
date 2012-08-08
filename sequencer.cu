#include "printFunctions.cu"
#include <time.h>

char * copySequenceToDevice (char ** sequences, int numSequences, int sequenceLength) {
  char * d_sequences;
  cudaMalloc (&d_sequences, sizeof (char) * sequenceLength * numSequences);

  for (int i = 0; i < numSequences; i++)
    cudaMemcpy (d_sequences + i * sequenceLength, *(sequences + i), sizeof (char) * sequenceLength, cudaMemcpyHostToDevice); 

  return d_sequences;
}

__global__ void createBuckets (char * sequence, char * buckets, int sequenceLength, int numBuckets, int matchLength, int blockBuckets) {
  //  int bucketIndex = threadIdx.x * matchLength * (1 + blockIdx.x % blockBuckets);
  // int bucketIndex = (threadIdx.x + blockIdx.x * threadIdx.x) * (1 + blockIdx.x % blockBuckets);
  // int sequenceIndex = (threadIdx.x + blockIdx.x * threadIdx.x) * (1 + blockIdx.x % blockBuckets);
  if (threadIdx.x == 0 && blockIdx.x == 0) 
   printf("kernel2\n");
  int index = threadIdx.x + (blockIdx.x % blockBuckets * blockDim.x);
  // if (threadIdx.x == 1)
  //   printf ("index = %d\n", index);
  if (index < numBuckets)
    for (int i = 0; i < matchLength; i++)
      *(buckets + index * matchLength + i) = *(sequence + index + i);
}

__global__ void assignBuckets (char * sequences, char * buckets, uint * bucketCounts, int numSequences, int sequenceLength, int numBuckets, int matchLength, double matchAccuracy, int blockBuckets) {

  // printf ("now testing sequence %d, bucket %d\n", blockIdx.x, threadIdx.x);

  // read sequence into shared memory for faster access
  extern __shared__ char sharedSequence[];  
  for (int i = threadIdx.x; i < sequenceLength; i += blockDim.x) 
    sharedSequence[i] = sequences[blockIdx.x / blockBuckets * sequenceLength + i];

  syncthreads();
  
  int numMatches = 0;
  int bucketIndex = threadIdx.x * matchLength * (1 + blockIdx.x % blockBuckets);
  //  if (threadIdx.x == 0 && blockIdx.x == 0) 
  //  printf("kernel\n");

  //  for (int i = 0; i < matchLength; i++)
  //   printf("%c", *(buckets + bucketIndex + i));
  // printf("%c", *(sharedSequence + i));

  //   printf ("\n");
  //  printf("blockBuckets = %d\n", blockBuckets);
  //}

  //if (bucketIndex < (numBuckets - 1) * matchLength)  
  // to go through entire sequence, up to sequenceLength - matchLength + 1
  for (int i = 0; i < numBuckets; i++) {
    // check if this thread's bucket matches the current sequence matchLength segment
    if ((bucketIndex < numBuckets * (matchLength - 1)) & (i < sequenceLength - matchLength)) {
      for (int j = 0; j < matchLength; j++) {
        if (threadIdx.x == 1023)
          printf("thread = %d\tblock=%d\tbucketIndex = %d,\tsequenceIndex = %d\n", threadIdx.x, blockIdx.x, bucketIndex +j, i + j);
        if (*(buckets + bucketIndex + j) == *(sharedSequence + i + j))
            numMatches++;       
      }
      // if (numMatches >= 20);
      // if (numMatches / (float) matchLength >= matchAccuracy) ;
      // printf ("%d\n", bucketIndex / (float) matchLength);
      //  printf("x\n");
      //   atomicInc (bucketCounts + (bucketIndex / matchLength), UINT_MAX);
      // break;
      //  }
      //   if (threadIdx.x % 2)
      //    printf("diverge\n");
      numMatches = 0;    
    }
  }
  
  // for debugging
  // atomicInc (bucketCounts + numBuckets, UINT_MAX);
}


uint * sequencer (char * d_sequences, int numSequences, int sequenceLength, int matchLength, double matchAccuracy) {
  // printf("running sequences\n");


  // printSequences (sequences, numSequences, sequenceLength);
  // printDeviceFirstLast (d_sequences, numSequences, sequenceLength);

  // choose a random sequence to create buckets from
  srand (time (NULL));
  int bucketSequence = 0;//rand() % numSequences;
  
  // printf ("bucketSequence = %d\n", bucketSequence);

  // create the buckets
  char * d_buckets;
  int numBuckets = sequenceLength - matchLength + 1;
  cudaMalloc (&d_buckets, sizeof (char) * numBuckets * matchLength); 


  int numThreads = 1024;
  if (numThreads > numBuckets)
    numThreads = numBuckets;

  int blockBuckets = ceil (numBuckets / (float) numThreads);
  int numBlocks = blockBuckets;

  createBuckets<<<numBlocks, numThreads>>> (d_sequences + bucketSequence * sequenceLength, d_buckets, numBuckets, sequenceLength, matchLength, blockBuckets);

  // make counters for each bucket, with the last one counting how many didn't fit
  //  into any buckets
  uint * d_bucketCounts;
  cudaMalloc (&d_bucketCounts, sizeof (uint) * (numBuckets + 1));
  cudaMemset (d_bucketCounts, 0, sizeof (uint) * (numBuckets + 1));
  cudaThreadSynchronize();
  // printDeviceSequences (d_buckets, numBuckets, matchLength);

  printf("numBlocks = %d, numThreads = %d, sharedmemsize = %d\n", numBlocks * numSequences, numThreads, sizeof (char) * sequenceLength);

  // assign the buckets
  assignBuckets<<<numBlocks * numSequences, numThreads, sizeof (char) * sequenceLength>>> (d_sequences, d_buckets, d_bucketCounts, numSequences, sequenceLength, numBuckets, matchLength, matchAccuracy, blockBuckets);
  // err = cudaGetLastError();
  //if (err != cudaSuccess) 
  //  printf("Error: %s\n", cudaGetErrorString(err));

  cudaThreadSynchronize();

  uint * bucketCounts = (uint *) malloc (sizeof (uint) * (numBuckets + 1));
  cudaMemcpy (bucketCounts, d_bucketCounts, sizeof (uint) * (numBuckets + 1), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < numBuckets + 1; i++)
  // printf ("bucketCounts[%d] = %u\n", i, *(bucketCounts + i));
  

  // printDeviceSequences (d_buckets, numBuckets, matchLength);

  // run kernel in loop from length of sequence down to ~10 or so to see
  //  which bucket sizes give good results
  //  will need an array which holds what the matching pattern is
  //  will need an array to store data of which sequences have matching pattern
  
  cudaFree (d_bucketCounts);
  cudaFree (d_buckets);
  free (bucketCounts);

  return NULL;
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
