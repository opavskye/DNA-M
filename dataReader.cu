#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int readSequences (char * fileName, char ** sequences, int numSequences, int sequenceLength) {

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

void printSequences (char ** sequences, int numSequences) {
  for (int i = 0; i < numSequences; i++)
    printf("sequence %d = %s\n", i, sequences[i]);
}

int main (int argc, char *argv[]) {
  
  char fileName[100];
  strcpy (fileName, "../data/");

  int numSequences = 125;
  int sequenceLength = 201;

  // allocate memory for sequences
  char ** sequences =  sequences = (char **) malloc (numSequences * sizeof (char *));
  for (int i = 0; i < numSequences; i++)
    *(sequences + i) = (char *) malloc (sequenceLength * sizeof (char));


  if (argc >= 1) 
    { 
      // append argv[1] to the end of path to data folder
      int i = 0; 
      while ((fileName[(i++) + 8] = argv[1][i]) != '\0');
    } 
  else
    {
      printf ("Please run again with a filename input from the data folder.\n");
      return 1;
    }

  // read in the data
  if (!readSequences (fileName, sequences, numSequences, sequenceLength))
    printf ("error reading data\n");

  printSequences (sequences, numSequences);

  // free all allocated memory
  for (int i = 0; i < numSequences; i++)
    free (sequences[i]);
  free (sequences);

  return 0;
}
