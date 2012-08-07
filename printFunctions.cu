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

void printDeviceFirstLast (char * d_sequences, int numSequences, int sequenceLength) {
  char * temp = (char *) malloc (sizeof (char) * 2 * sequenceLength);

  cudaMemcpy (temp, d_sequences, sizeof (char) * sequenceLength, cudaMemcpyDeviceToHost);
  cudaMemcpy (temp + sequenceLength, d_sequences + sequenceLength * (numSequences - 1), sizeof (char) * sequenceLength, cudaMemcpyDeviceToHost);

  int i;

  printf ("d_sequences[0] = ");
  for (i = 0; i < sequenceLength; i++)
    printf ("%c", *(temp + i));
  printf ("\n");

  printf ("d_sequences[%d] = ", numSequences - 1);
  for (; i < sequenceLength * 2; i++)
    printf ("%c", *(temp + i));
  printf ("\n");

  free (temp);
}


void printFirstLast (char ** sequences, int numSequences, int sequenceLength) {
  printf("sequences[0] = %s\n", sequences[0]);
  printf("sequences[%d] = %s\n", numSequences - 1, sequences[numSequences - 1]);
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
