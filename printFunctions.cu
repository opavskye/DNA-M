#include <stdio.h>

void printDeviceSequences (char * d_sequences, int numSequences, int sequenceLength) {
  char * temp = (char *) malloc (sizeof (char) * numSequences * sequenceLength);

  cudaMemcpy (temp, d_sequences, sizeof (char) * numSequences * sequenceLength, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < numSequences * sequenceLength; i += sequenceLength)
  //  printf ("d_sequences[%d] = %s\n", i / sequenceLength, temp + i);  
  for (int i = 0; i < numSequences; i++) {
    printf ("d_sequences[%d] = ", i);
    for (int j = 0; j < sequenceLength; j++)
      printf ("%c", *(temp + i * sequenceLength + j));  
    printf ("\n");
  }

  free (temp);
}

void printSequences (char ** sequences, int numSequences, int sequenceLength) {
  for (int i = 0; i < numSequences; i++) {
    printf ("sequences[%d] = ", i);
    for (int j = 0; j < sequenceLength; j++)
      printf ("%c", sequences[i][j]);
    printf ("\n");
  }
}

/*
  void printFlatSequences (char * sequences, int numSequences, int sequenceLength) {
  for (int i = 0; i < numSequences; i++)
  printf ("flat_sequences[%d] = %s\n", i , sequences + i * sequenceLength);  
  }
*/
