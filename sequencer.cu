#include "printFunctions.cu"
#include "maximums.c"

#define THREADS_PER_BLOCK 1024
#define OUTPUTS_TO_KEEP 5


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

__global__ void assignBuckets (char * sequence, char * bucketSequence, uint * tempBucketCounts, int sequenceLength, int numBuckets, int matchLength, double matchAccuracy, int numSequenceSections) {
 
  extern __shared__ char shared[];
  char * sharedSequence = &shared[0];
  char * sharedBucketSequence = &shared[blockDim.x + matchLength];

  // start of this current sequence section
  int sequenceIndex = blockIdx.y * blockDim.x;

  if (threadIdx.x + sequenceIndex < sequenceLength) 
    *(sharedSequence + threadIdx.x) = *(sequence + sequenceIndex + threadIdx.x);
  
  if (threadIdx.x < matchLength) {
    *(sharedSequence + blockDim.x + threadIdx.x) = *(sequence + sequenceIndex + blockDim.x + threadIdx.x);
    *(sharedBucketSequence + threadIdx.x) = *(bucketSequence + blockIdx.x + threadIdx.x);
  }

  syncthreads();

  int numMatches = 0;
  int index;
  
  for (int i = 0; i < matchLength; i++) 
    if ((index = threadIdx.x + i) + sequenceIndex < sequenceLength)
      if (*(sharedSequence + index) == *(sharedBucketSequence + i) && *(sharedSequence + index) != 'N')
        numMatches++;

  if ((numMatches / (double) matchLength >= matchAccuracy) && (sequenceIndex + threadIdx.x < numBuckets)) 
    // blockIdx.x * numBuckets is which bucket in tempBucketCounts
    // sequenceIndex + threadIdx.x is which part of the sequence was just checked
    atomicInc (tempBucketCounts + blockIdx.x * numBuckets + sequenceIndex + threadIdx.x, UINT_MAX);
}

__global__ void transferBuckets (uint * buckets, uint * tempBuckets, int numBuckets) {

  int bucketIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (bucketIndex < numBuckets)
    for (int i = 0; i < numBuckets; i++) {
      if (*(tempBuckets + bucketIndex * numBuckets + i)) {
        (*(buckets + bucketIndex))++;
        return;
      }
    }
}

__global__ void counterKernel (char * sequences, int sequenceLength, char * query, int queryLength, uint * count, double matchAccuracy) {

  // read query and sequence segment into shared memory for faster access
  extern __shared__ char shared[];
  char * sharedQuery = &shared[0];
  char * sharedSequence = &shared[queryLength];

  // start of current sequence section
  int sequenceIndex = blockIdx.x * sequenceLength + blockIdx.y * blockDim.x + threadIdx.x;

  if (sequenceIndex < sequenceLength)
    *(sharedSequence + threadIdx.x) = *(sequences + sequenceIndex);

  if (threadIdx.x < queryLength) {
    *(sharedQuery + threadIdx.x) = query[threadIdx.x];
    *(sharedSequence + blockDim.x + threadIdx.x) = *(sequences + sequenceIndex + blockDim.x);
  }

  int numMatches = 0;

  for (int i = 0; i < queryLength; i++) {
    if (*(sequences + sequenceIndex + i) == *(query + i))
      numMatches++;
  }

  if (numMatches / (double) queryLength >= matchAccuracy)
    atomicInc (count + blockIdx.x * (sequenceLength - queryLength + 1) + blockIdx.y * blockDim.x + threadIdx.x, UINT_MAX);
}

bucketData sequencer (char * d_sequences, int numSequences, int sequenceLength, int bucketSequence, int matchLength, double matchAccuracy) {

  // printSequences (sequences, numSequences, sequenceLength);
  // printDeviceSequences (d_sequences, numSequences, sequenceLength);
	  
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
  
  //printf ("numThreads = %d, numBlocks = %d, numShared = %d, numBuckets = %d, bucketsPerThread = %d\n", numThreads, numSequences, sequenceLength * 2, numBuckets, bucketsPerThread);
  
  /*
  printFirstLastBuckets (d_bucketSequence, numBuckets, matchLength, sequenceLength);
  printDeviceFirstLast (d_sequences, numSequences, sequenceLength);
  */
  
  numThreads = THREADS_PER_BLOCK;
  if (sequenceLength < numThreads)
    numThreads = sequenceLength;

  int numSequenceSections = ceil (sequenceLength / (float) numThreads);

  // make counters for each bucket in each sequence
  uint * d_bucketCounts;
  cudaMalloc (&d_bucketCounts, sizeof (uint) * numBuckets);
  cudaMemset (d_bucketCounts, 0, sizeof (uint) * numBuckets);

  uint * d_tempBucketCounts;
  cudaMalloc (&d_tempBucketCounts, sizeof (uint) * numBuckets * numBuckets);

  
  // printf ("numThreads = %d, numBlocks = %d, numBuckets = %d, numSequenceSections = %d\n", numThreads, numSequenceSections * numBuckets, numBuckets, numSequenceSections);


  // analyze 1 sequence at a time because executing one kernel for all the data
  //  in a large data set takes too long and causes watchdog error
  dim3 gridDim (numBuckets, numSequenceSections);
  for (int i = 0; i < numSequences; i++) {
    cudaMemset (d_tempBucketCounts, 0, sizeof (uint) * numBuckets * numBuckets);
    assignBuckets<<<gridDim, numThreads, sizeof (char) * (numThreads + matchLength * 2)>>> (d_sequences + i * sequenceLength, d_bucketSequence, d_tempBucketCounts, sequenceLength, numBuckets, matchLength, matchAccuracy, numSequenceSections);

    // write kernel for d_bucketCounts which makes it account for repeats of a sequence segment
    //  this kernel won't work if numSequenceSections > 1024, but that will only happen
    //  if sequenceLength > 1024 * 1024, which is not relevent at this time
    transferBuckets<<<bucketsPerThread, numThreads>>> (d_bucketCounts, d_tempBucketCounts, numBuckets);
    cudaThreadSynchronize();
  }
  
  /*
  printf("\nnow printing after assignBuckets:\n\n");
  printFirstLastBuckets (d_bucketSequence, numBuckets, matchLength, sequenceLength); 
  printDeviceFirstLast (d_sequences, numSequences, sequenceLength);
  */  
  
  uint * bucketCounts = (uint *) malloc (sizeof (uint) * numBuckets);
  cudaMemcpy (bucketCounts, d_bucketCounts, sizeof (uint) * numBuckets, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < numBuckets + 1; i++)
  // printf ("bucketCounts[%d] = %u\n", i, *(bucketCounts + i));

  // fill data with top OUTPUTS_TO_KEEP number of top bucketCounts 
  bucketData data;
  int maxes[OUTPUTS_TO_KEEP];
  findMaximums (bucketCounts, numBuckets, maxes, OUTPUTS_TO_KEEP);

  for (int i = 0; i < OUTPUTS_TO_KEEP; i++) {
    data.sequenceIndex[i] = bucketSequence;
    data.bucketNum[i] = maxes[i];
    data.count[i] = bucketCounts[maxes[i]];

    cudaMemcpy (data.bucketContents[i], d_sequences + bucketSequence * sequenceLength + maxes[i], sizeof (char) * matchLength, cudaMemcpyDeviceToHost);
    data.bucketContents[i][matchLength] = '\0';
  }

  cudaFree (d_bucketCounts);
  cudaFree (d_bucketSequence);

  return data;
}


// grep -c query fileName
uint counter (char * d_sequences, int numSequences, int sequenceLength, char * query, int queryLength, double matchAccuracy) {

  // put query into device memory
  char * d_query;
  cudaMalloc (&d_query, queryLength * sizeof (char));
  cudaMemcpy (d_query, query, queryLength * sizeof (char), cudaMemcpyHostToDevice);

  int numBuckets = sequenceLength - queryLength + 1;
  int numThreads = numBuckets;
  if (numThreads > THREADS_PER_BLOCK)
    numThreads = THREADS_PER_BLOCK;

  int numSequenceSections = ceil ((numBuckets) / (float) numThreads);

  uint * d_tempCounters;
  cudaMalloc (&d_tempCounters, numSequences * numBuckets * sizeof (int));
  cudaMemset (d_tempCounters, 0, sizeof (uint) * numSequences * numBuckets);

  dim3 gridDim (numSequences, numSequenceSections);
  counterKernel<<<gridDim, numThreads, (queryLength * 2 + numThreads) * sizeof (char)>>> (d_sequences, sequenceLength, d_query, queryLength, d_tempCounters, matchAccuracy);

  uint count = 0;
  uint * tempCounters = (uint *) malloc (sizeof (uint) * numSequences * numBuckets);
  cudaMemcpy (tempCounters, d_tempCounters, sizeof (uint) * numSequences * numBuckets, cudaMemcpyDeviceToHost);

  for (int i = 0; i < numSequences; i++)
    for (int j = 0; j < numBuckets; j++) 
      if (tempCounters[i * numBuckets + j]) {
        count++;
        break;
      }  

  cudaFree (d_tempCounters);
  cudaFree (d_query);

  return count;
}
