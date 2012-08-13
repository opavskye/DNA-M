#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>

#include "sequencer.cu"

typedef struct {
  int sequenceIndex;
  int bucketNum;
  char bucketContents[21];
  int count;
} bucketData;

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
  // int matchLength = 20;
  double matchAccuracy = .9;

  char * outFile;

  // char * s1 = "CATTCAGCTTGCACTTTGGA";
  // char * s2 = "CAGGCAGCTC";
  // char * s3 = "CTAACTGGGG";

  if (argc == 3)
    outFile = argv[1];
    
  if (argc == 2) {
    outFile = argv[1];

    fileEnd[0] = 'r';
    fileEnd[1] = 'a';
    fileEnd[2] = 'n';
    fileEnd[3] = 'd';
    fileEnd[4] = 'o';
    fileEnd[5] = 'm';
  }

  /*
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
  */

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
  uint ** results = (uint **) malloc ((maxLength - minLength + 1) * sizeof (uint **));
  uint maximums[maxLength - minLength + 1];
  int maxIndices[maxLength - minLength + 1];
  
  FILE * out;
  out = fopen (outFile, "w");

  fprintf (out, "file name = %s, numSequences = %d, sequenceLength = %d\n", fileEnd, numSequences, sequenceLength);

  for (int i = minLength; i <= maxLength; i++) {
    fprintf (out, "\n\nNow running matchLength = %d\n\n", i);
    for (int bucketSequence = 0; bucketSequence < numSequences; bucketSequence += 5) {
      results[i - minLength] = sequencer (d_sequences, numSequences, sequenceLength, bucketSequence, i, matchAccuracy);
      maxIndices[i - minLength] = maximum (results[i - minLength], sequenceLength - i + 1);
      maximums[i - minLength] = results[i - minLength][maxIndices[i - minLength]];
      fprintf (out, "There were maximum %u matching sequences at bucket %d.\n", maximums[i - minLength], maxIndices[i - minLength]);
    
      fprintf (out, "Bucket[%d] of sequence[%d] = ", maxIndices[i - minLength], bucketSequence);
      for (int j = maxIndices[i - minLength]; j < maxIndices[i - minLength] + i; j++)
        fprintf(out, "%c", sequences[bucketSequence][j]);
      fprintf(out, "\n");

      /*
        for (int j = 0; j < sequenceLength - i + 1; j++) {
        printf("bucket %d; found %d instances of string ", j, results[i - minLength][0]);
        for (int k = 0; k < i; k++)
        printf("%c", sequences[0][j + k]);
        printf("\n");
        }
      */
      // printf ("printing device firstlast now\n");
      // printDeviceFirstLast (d_sequences, numSequences, sequenceLength);
    }
    fprintf(out, "\n\n");
  }

  fclose (out);

  // printf ("counter 1 = %u\n", counter (sequences, numSequences, sequenceLength, s1, 20, .9));
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
