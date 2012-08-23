/* Copyright 2012 by Erik Opavsky
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda.h>
#include <curand.h>

#include "dataTransfer.cu"
#include "counter.cu"

int main (int argc, char *argv[]) {
  
  if (argc != 4) {
    printf ("Argument Error:  correct usage is\t%s inputFile querySequence matchAccuracy\n", argv[0]);
    return 1;
  }

  char fileName[100];
  strcpy (fileName, "../data/");
  
  char * fileEnd = argv[1];
  char * query = argv[2];
  double matchAccuracy = atof (argv[3]);
  int numSequences = 129;
  int sequenceLength = 2000;

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

  printf ("file name = %s, query = %s, numSequences = %d, sequenceLength = %d, match threshold = %.2lf\n", fileEnd, query, numSequences, sequenceLength, matchAccuracy);



  // run counter
  printf ("%s counter = %u\n", query, counter (d_sequences, numSequences, sequenceLength, query, strlen (query), matchAccuracy));




  // free all allocated memory
  cudaFree (d_sequences);

  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  return 0;
}
