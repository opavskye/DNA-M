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

uint maximum (uint * list, int listLength) {
  uint max = list[0];

  for (int i = 1; i < listLength; i++)
    if (list[i] > max)
      max = list[i];

  return max;
}

int main (int argc, char *argv[]) {
  
  char fileName[100];
  strcpy (fileName, "../data/");

  char fileEnd[100] = "sample.csv";
  int numSequences = 125;
  int sequenceLength = 200;
  int matchLength = 10;
  double matchAccuracy = .9;

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

  int minLength = 4;
  int maxLength = 20;
  uint ** results = (uint **) malloc ((maxLength - minLength + 1) * sizeof (uint **));
  uint maximums[maxLength - minLength + 1];

  for (int i = minLength; i <= maxLength; i++) {
    results[i - minLength] = sequencer (sequences, numSequences, sequenceLength, i, matchAccuracy);
    maximums[i - minLength] = maximum (results[i - minLength], sequenceLength - i + 1);
    printf("For matchLength = %d, there were maximum %u matching sequences.\n", i, maximums[i - minLength]);
  }
  

  //  char * s1 = "AGAGTTGTGG";
  //  char * s2 = "CAGGCAGCTC";
  //  char * s3 = "CTAACTGGGG";

  // printf ("counter 1 = %u\n", counter (sequences, numSequences, sequenceLength, s1, 10, .8));
  // printf ("counter 2 = %u\n", counter (sequences, numSequences, sequenceLength, s2, 10, .8));
  // printf ("counter 3 = %u\n", counter (sequences, numSequences, sequenceLength, s3, 10, .8));

  // free all allocated memory
  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  for (int i = minLength; i <= maxLength; i++)
    free (results[i - minLength]);
  free (results);

  return 0;
}
