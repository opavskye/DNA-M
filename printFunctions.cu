#include <stdio.h>

void printDeviceSequences (char * d_sequences, int numSequences, int sequenceLength) {
  char * temp = (char *) malloc (sizeof (char) * numSequences * sequenceLength);

  cudaMemcpy (temp, d_sequences, sizeof (char) * numSequences * sequenceLength, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < numSequences * sequenceLength; i += sequenceLength)
  //  printf ("d_sequences[%d] = %s\n", i / sequenceLength, temp + i);  
   for (int i = 0; i < numSequences; i++)
    printf ("d_sequences[%d] = %s\n", i , temp + i * sequenceLength);  

  free (temp);
}

void printSequences (char ** sequences, int numSequences) {
  for (int i = 0; i < numSequences; i++)
    printf("sequences[%d] = %s\n", i, *(sequences + i));
}

void printFlatSequences (char * sequences, int numSequences, int sequenceLength) {
   for (int i = 0; i < numSequences; i++)
     printf ("flat_sequences[%d] = %s\n", i , sequences + i * sequenceLength);  
}
