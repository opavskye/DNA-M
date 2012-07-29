#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

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


int main (int argc, char *argv[]) {
  
  char fileName[100];
  strcpy (fileName, "../data/");

  char fileEnd[100] = "sample.csv";
  int numSequences = 125;
  int sequenceLength = 200;
  int matchLength = 5;
  double matchAccuracy = 1;

  if (argc < 2) {
    printf ("Please enter the name of the data file: ");
    scanf ("%s", fileEnd);

    printf ("Please enter the number of sequences: ");
    scanf ("%d", &numSequences);

    printf ("Please enter the length of the sequences: ");
    scanf ("%d", &sequenceLength);

    printf ("Please enter the length of the matching substrings: ");
    scanf ("%d", &matchLength);

    printf ("Please enter the minimum accuracy of the matches: ");
    scanf ("%lf", &matchAccuracy);
  }


  // allocate memory for sequences
  char ** sequences =  sequences = (char **) malloc (numSequences * sizeof (char *));
  for (int i = 0; i < numSequences; i++)
    *(sequences + i) = (char *) malloc ((sequenceLength + 1) * sizeof (char));

  // append fileEnd to the end of path to data folder
  int i = 0; 
  while ((fileName[(i++) + 8] = fileEnd[i]) != '\0');

  // read in the data
  if (!readSequences (fileName, sequences, numSequences))
    printf ("error reading data\n");

  // printSequences (sequences, numSequences);

  sequencer (sequences, numSequences, sequenceLength, matchLength, matchAccuracy);

  // free all allocated memory
  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  return 0;
}
