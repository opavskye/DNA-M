/* Copyright 2012 by Erik Opavsky
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "constants.h"

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
