#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>

#include "sequencer.cu"
#include "dataTransfer.cu"

int main (int argc, char *argv[]) {
  

  if (argc != 4) {
    printf ("Argument Error:  correct usage is\t./sequencer inputFile outputFile matchAccuracy\n");
    return 1;
  }

  char inFile[100];
  strcpy (inFile, "../data/");
  
  char * fileEnd = argv[1];
  char * outFile = argv[2];
  double matchAccuracy = atof (argv[3]);
  int numSequences = 1690;
  int sequenceLength = 2000;
  int minLength = 4;
  int maxLength = 15;


  // allocate memory for sequences
  char ** sequences = (char **) malloc (numSequences * sizeof (char *));
  for (int i = 0; i < numSequences; i++)
    *(sequences + i) = (char *) malloc ((sequenceLength + 1) * sizeof (char));

  // append fileEnd to the end of path to data folder
  int i = 0; 
  while ((inFile[(i++) + 8] = fileEnd[i]) != '\0');

  // read in the data
  if (!readSequences (inFile, sequences, numSequences)) {
    printf ("error reading data\n");
    return 1;
  }

  // put sequences into device memory
  char * d_sequences = copySequencesToDevice (sequences, numSequences, sequenceLength);

  // make sure data was read in correctly
  // printDeviceFirstLast (d_sequences, numSequences, sequenceLength);

  
  // choose a random sequence to create buckets from
  // srand (time (NULL));
  // int bucketSequence = 1;//rand() % numSequences;

  FILE * out;
  out = fopen (outFile, "a");

  fprintf (out, "frequency,consensus,threshold,file name\n");

  fclose (out);

  // fprintf (out, "file name = %s, numSequences = %d, sequenceLength = %d, match threshold = %.2lf\n", fileEnd, numSequences, sequenceLength, matchAccuracy);
  // printf ("file name = %s, numSequences = %d, sequenceLength = %d, match threshold = %.2lf\n", fileEnd, numSequences, sequenceLength, matchAccuracy);
  for (int i = minLength; i <= maxLength; i++) {
    bucketData results[numSequences];

    // fprintf (out, "\n\nNow running matchLength = %d\n\n", i);
    // printf ("\n\nNow running matchLength = %d\n\n", i);
    for (int bucketSequence = 0; bucketSequence < numSequences; bucketSequence ++) {
      results[bucketSequence] = sequencer (d_sequences, numSequences, sequenceLength, bucketSequence, i, matchAccuracy);
     
      printf ("bucketSequence %d for length %d complete.\n", bucketSequence, i);
      
      /*
        for (int x = 0; x < OUTPUTS_TO_KEEP; x++) {
        printf ("There were maximum %u matching sequences at bucket %d.\n", results[bucketSequence].count[x], results[bucketSequence].bucketNum[x]);
        printf ("sequence = %s\n\n", results[bucketSequence].bucketContents[x]);
        }
      */ 
    }
  
    printf ("now summarizing maximums\n");
    bucketData maxBucket = summarizeMaximums (results, numSequences, OUTPUTS_TO_KEEP);

    printf ("now printing outputs\n");

    out = fopen (outFile, "a");

    for (int x = 0; x < OUTPUTS_TO_KEEP; x++) {
      printf ("now writing to file:  %d,%s,%.2lf,%s\n", maxBucket.count[x], maxBucket.bucketContents[x], matchAccuracy, fileEnd);
      fprintf (out, "%d,%s,%.2lf,%s\n", maxBucket.count[x], maxBucket.bucketContents[x], matchAccuracy, fileEnd);
      printf ("MAX:\tcount = %u, bucket = %d, sequence = %d, contents = %s\n\n", maxBucket.count[x], maxBucket.bucketNum[x], maxBucket.sequenceIndex[x], maxBucket.bucketContents[x]);
    }

    fclose (out);
  }

  // free all allocated memory
  cudaFree (d_sequences);

  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  return 0;
}
