#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#include "sequencer.cu"

int readSequences (char * fileName, char ** sequences, int numSequences) {

  FILE *dataFile;
  if ((dataFile = fopen (fileName, "r")) == NULL) {
    printf("The file %s could not be opened.\n", fileName);
    return 0;
  }

  while (getc (dataFile) != ',');
  for (int i = 0; i < numSequences; i++) {

    // skip first column
    while (getc (dataFile) != ',');
    fscanf (dataFile, "%s", sequences[i]);

  }

  fclose (dataFile);
  return 1;
}

int main() {

  int numSequences = 2001;
  int sequenceLength = 2000;
  int minLength = 4;
  int maxLength = 20;
  
  char * fileName = "../data/sample3.csv";

  // allocate memory for sequences
  char ** sequences =  sequences = (char **) malloc (numSequences * sizeof (char *));
  for (int i = 0; i < numSequences; i++)
    *(sequences + i) = (char *) malloc ((sequenceLength + 1) * sizeof (char));

  // read in the data
  if (!readSequences (fileName, sequences, numSequences)) {
    printf ("error reading data\n");
    return 1;
  }

  uint ** results = (uint **) malloc ((maxLength - minLength + 1) * sizeof (uint **));

  // put sequences into device memory
  char * d_sequences = copySequenceToDevice (sequences, numSequences, sequenceLength);

  // printFirstLast (sequences, numSequences, sequenceLength);
  // printDeviceFirstLast (d_sequences, numSequences, sequenceLength);

  // works fine to here
  int matchLength = maxLength;
  int bucketSequence = 0;
  double matchAccuracy = 1;

  char * d_buckets;
  int numBuckets = sequenceLength - matchLength + 1;
  cudaMalloc (&d_buckets, sizeof (char) * numBuckets * matchLength); 

  uint * d_bucketCounts;
  cudaMalloc (&d_bucketCounts, sizeof (uint) * (numBuckets + 1));
  cudaMemset (d_bucketCounts, 0, sizeof (uint) * (numBuckets + 1));

  int numThreads = 1024;
  if (numThreads > numBuckets)
    numThreads = numBuckets;

  int blockBuckets = ceil (numBuckets / (float) numThreads);
  int numBlocks = blockBuckets;


  createBuckets<<<numBlocks, numThreads>>> (d_sequences + bucketSequence * sequenceLength, d_buckets, numBuckets, sequenceLength, matchLength, blockBuckets);
  cudaThreadSynchronize();

  printDeviceFirstLast (d_sequences, numSequences, sequenceLength);
  printDeviceFirstLast (d_buckets, numBuckets, matchLength);

  // assign the buckets
   assignBuckets<<<numBlocks * numSequences, numThreads, sizeof (char) * sequenceLength>>> (d_sequences, d_buckets, d_bucketCounts, numSequences, sequenceLength, numBuckets, matchLength, matchAccuracy, blockBuckets);

  //printFirstLast (sequences, numSequences, sequenceLength);
  printDeviceFirstLast (d_sequences, numSequences, sequenceLength);
  printDeviceFirstLast (d_buckets, numBuckets, matchLength);

  printf("numBlocks = %d, numThreads = %d, sharedmemsize = %d\n", numBlocks * numSequences, numThreads, sizeof (char) * sequenceLength);


  // free all allocated memory
  cudaFree (d_sequences);
  cudaFree (d_buckets);
  cudaFree (d_bucketCounts);

  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  free (results);

  return 0;
}
