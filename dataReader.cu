#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

  // skip first row
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
  double matchAccuracy = .8;

  char * outFile;

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
  

  /*
  for (int i = 1; i < argc; i++)
    printf ("%s counter = %u\n", argv[i], counter (d_sequences, numSequences, sequenceLength, argv[i], 4, matchAccuracy));
  */

  // free all allocated memory
  cudaFree (d_sequences);

  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  return 0;
}
