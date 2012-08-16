#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda.h>
#include <curand.h>

#include "dataTransfer.cu"
#include "counter.cu"

int main (int argc, char *argv[]) {
  
  if (argc != 4) {
    printf ("Argument Error:  correct usage is\t./counter inputFile querySequence matchAccuracy\n");
    return 1;
  }

  char fileName[100];
  strcpy (fileName, "../data/");
  
  char * fileEnd = argv[1];
  char * query = argv[2];
  double matchAccuracy = atof (argv[3]);
  int numSequences = 129;
  int sequenceLength = 2000;

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

  printf ("file name = %s, query = %s, numSequences = %d, sequenceLength = %d, match threshold = %.2lf\n", fileEnd, query, numSequences, sequenceLength, matchAccuracy);



  // run counter
  printf ("%s counter = %u\n", query, counter (d_sequences, numSequences, sequenceLength, query, strlen (query), matchAccuracy));




  // free all allocated memory
  cudaFree (d_sequences);

  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  return 0;
}
