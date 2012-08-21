#include <stdio.h>


int readSequences (char * fileName, char ** sequences, int numSequences) {

  cudaDeviceReset();

  FILE *dataFile;
  if ((dataFile = fopen (fileName, "r")) == NULL) {
    printf("The file %s could not be opened.\n", fileName);
    return 0;
  }

  // skip first row
  // while (getc (dataFile) != ',');

  for (int i = 0; i < numSequences; i++) {

    // skip first column
    while (getc (dataFile) != ',');

    // skip second column
    while (getc (dataFile) != ',');

    // skip third column
    // while (getc (dataFile) != ',');

    fscanf (dataFile, "%s", sequences[i]);

  }

  fclose (dataFile);
  return 1;
}

char * copySequencesToDevice (char ** sequences, int numSequences, int sequenceLength) {
  char * d_sequences;
  cudaMalloc (&d_sequences, sizeof (char) * sequenceLength * numSequences);

  for (int i = 0; i < numSequences; i++)
    cudaMemcpy (d_sequences + i * sequenceLength, *(sequences + i), sizeof (char) * sequenceLength, cudaMemcpyHostToDevice);

  return d_sequences;
}
