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

  char fileName[100];
  strcpy (fileName, "../data/");
  
  char * fileEnd = argv[1];

  // char fileEnd[100] = "sample.csv";
  int numSequences = 125;
  int sequenceLength = 200;
  // double matchAccuracy = .8;
  double matchAccuracy = atof (argv[3]);

  char * outFile = argv[2];

  // allocate memory for sequences
  char ** sequences =  sequences = (char **) malloc (numSequences * sizeof (char *));
  for (int i = 0; i < numSequences; i++)
    *(sequences + i) = (char *) malloc ((sequenceLength + 1) * sizeof (char));

  // append fileEnd to the end of path to data folder
  int i = 0; 
  while ((fileName[(i++) + 8] = fileEnd[i]) != '\0');

  // read in the data
  if (!readSequences (fileName, sequences, numSequences)) {
    printf ("error reading data\n");
    return 1;
  }

  // put sequences into device memory
  char * d_sequences = copySequencesToDevice (sequences, numSequences, sequenceLength);


  
  // choose a random sequence to create buckets from
  srand (time (NULL));
  // int bucketSequence = 1;//rand() % numSequences;
  int minLength = 4;
  int maxLength = 20;

  FILE * out;
  out = fopen (outFile, "w");

  fprintf (out, "file name = %s, numSequences = %d, sequenceLength = %d, match threshold = %lf\n", fileEnd, numSequences, sequenceLength, matchAccuracy);
  printf ("file name = %s, numSequences = %d, sequenceLength = %d, match threshold = %lf\n", fileEnd, numSequences, sequenceLength, matchAccuracy);
  for (int i = minLength; i <= maxLength; i++) {
    bucketData results[numSequences];

    fprintf (out, "\n\nNow running matchLength = %d\n\n", i);
    printf ("\n\nNow running matchLength = %d\n\n", i);
    for (int bucketSequence = 0; bucketSequence < numSequences; bucketSequence ++) {
      results[bucketSequence] = sequencer (d_sequences, numSequences, sequenceLength, bucketSequence, i, matchAccuracy);
     

      for (int x = 0; x < OUTPUTS_TO_KEEP; x++) {
        fprintf (out, "There were maximum %u matching sequences at bucket %d.\n", results[bucketSequence].count[x], results[bucketSequence].bucketNum[x]);
        fprintf (out, "sequence = %s\n", results[bucketSequence].bucketContents[x]);
      }
    }

    fprintf(out, "\n\nSUMMARY FOR MATCHLENGTH = %d:\n", i);
  
    bucketData maxBucket = summarizeMaximums (results, numSequences, OUTPUTS_TO_KEEP);

    for (int x = 0; x < OUTPUTS_TO_KEEP; x++) {
      fprintf (out, "MAX:\tcount = %u, bucket = %d, sequence = %d, contents = %s\n", maxBucket.count[x], maxBucket.bucketNum[x], maxBucket.sequenceIndex[x], maxBucket.bucketContents[x]);
      printf ("MAX:\tcount = %u, bucket = %d, sequence = %d, contents = %s\n", maxBucket.count[x], maxBucket.bucketNum[x], maxBucket.sequenceIndex[x], maxBucket.bucketContents[x]);

    }
    fprintf (out, "\n\n");
  }

  fclose (out);
  

  
  //  for (int i = 1; i < argc; i++)
  //  printf ("%s counter = %u\n", argv[i], counter (d_sequences, numSequences, sequenceLength, argv[i], 4, matchAccuracy));
  

  // free all allocated memory
  cudaFree (d_sequences);

  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  return 0;
}