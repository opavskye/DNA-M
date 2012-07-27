#include "printFunctions.cu"


char * copySequenceToDevice (char ** sequences, int numSequences, int sequenceLength) {
  char * d_sequences;
  cudaMalloc (&d_sequences, sizeof (char) * sequenceLength * numSequences);

  for (int i = 0; i < numSequences; i++)
    cudaMemcpy (d_sequences + i * sequenceLength, *(sequences + i), sizeof (char) * sequenceLength, cudaMemcpyHostToDevice); 

  return d_sequences;
}

void sequencerWrapper (char ** sequences, int numSequences, int sequenceLength, double matchAccuracy) {

  // put sequences into device memory
  char * d_sequences = copySequenceToDevice (sequences, numSequences, sequenceLength);



  // run kernel in loop from length of sequence down to ~10 or so to see
  //  which bucket sizes give good results
  //  will need an array which holds what the matching pattern is
  //  will need an array to store data of which sequences have matching pattern
  

  cudaFree (d_sequences);

}
