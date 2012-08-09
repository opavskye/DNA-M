#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include "sequencer.cu"

int readSequences (char * fileName, char ** sequences, int numSequences) {

  cudaDeviceReset();

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

int maximum (uint * list, int listLength) {
  int max = 0;

  for (int i = 1; i < listLength; i++)
    if (list[i] > list[max])
      max = i;

  return max;
}

int main (int argc, char *argv[]) {
  
  char fileName[100];
  strcpy (fileName, "../data/");

  char fileEnd[100] = "sample.csv";
  int numSequences = 125;
  int sequenceLength = 200;
  int matchLength = 20;
  double matchAccuracy = 1;

  if (argc == 2) {
    fileEnd[6] = '2';
    fileEnd[7] = '.';
    fileEnd[8] = 'c';
    fileEnd[9] = 's';
    fileEnd[10] = 'v';
    fileEnd[11] = '\0';
    numSequences = 3000;
    sequenceLength = 4000;
  }  
  if (argc == 3) {
    fileEnd[6] = '3';
    fileEnd[7] = '.';
    fileEnd[8] = 'c';
    fileEnd[9] = 's';
    fileEnd[10] = 'v';
    fileEnd[11] = '\0';
    numSequences = 2001;
    sequenceLength = 2000;
  }

  if (argc == 4) {
    fileEnd[6] = '4';
    fileEnd[7] = '.';
    fileEnd[8] = 'c';
    fileEnd[9] = 's';
    fileEnd[10] = 'v';
    fileEnd[11] = '\0';
    numSequences = 1000;
    sequenceLength = 1000;
  }

  if (argc == 5) {
    fileEnd[6] = '5';
    fileEnd[7] = '.';
    fileEnd[8] = 'c';
    fileEnd[9] = 's';
    fileEnd[10] = 'v';
    fileEnd[11] = '\0';
    numSequences = 1000;
    sequenceLength = 2000;
  }

  if (argc == 6) {
    fileEnd[6] = '6';
    fileEnd[7] = '.';
    fileEnd[8] = 'c';
    fileEnd[9] = 's';
    fileEnd[10] = 'v';
    fileEnd[11] = '\0';
    numSequences = 4000;
    sequenceLength = 1000;
  }

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
  if (!readSequences (fileName, sequences, numSequences)) {
    printf ("error reading data\n");
    return 1;
  }

  // printFirstLast (sequences, numSequences, sequenceLength);

  int minLength = 4;
  int maxLength = 20;
  uint ** results = (uint **) malloc ((maxLength - minLength + 1) * sizeof (uint **));
  uint maximums[maxLength - minLength + 1];
  int maxIndices[maxLength - minLength + 1];

  // put sequences into device memory
  char * d_sequences = copySequencesToDevice (sequences, numSequences, sequenceLength);

  printf ("numSequences = %d, sequenceLength = %d\n", numSequences, sequenceLength);

  for (int i = minLength; i <= maxLength; i++) {
    results[i - minLength] = sequencer (d_sequences, numSequences, sequenceLength, i, matchAccuracy);
    maxIndices[i - minLength] = maximum (results[i - minLength], sequenceLength - i + 1);
    maximums[i - minLength] = results[i - minLength][maxIndices[i - minLength]];
    printf ("For matchLength = %d, there were maximum %u matching sequences at bucket %d.\n\n", i, maximums[i - minLength], maxIndices[i - minLength]);
    // printf ("printing device firstlast now\n");
    // printDeviceFirstLast (d_sequences, numSequences, sequenceLength);

  }
  

  // char * s1 = "AGAGTTGTGG";
  // char * s2 = "CAGGCAGCTC";
  // char * s3 = "CTAACTGGGG";

  // printf ("counter 1 = %u\n", counter (sequences, numSequences, sequenceLength, s1, 10, .8));
  // printf ("counter 2 = %u\n", counter (sequences, numSequences, sequenceLength, s2, 10, .8));
  // printf ("counter 3 = %u\n", counter (sequences, numSequences, sequenceLength, s3, 10, .8));

  // free all allocated memory
  cudaFree (d_sequences);

  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  for (int i = minLength; i <= maxLength; i++)
    free (results[i - minLength]);
  free (results);

  return 0;
}
