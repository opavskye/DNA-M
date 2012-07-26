#include <stdio.h>
#include <string.h>

int readSequences (char * fileName, char ** sequences, int numSequences, int sequenceLength) {

  FILE *dataFile;
  if ((dataFile = fopen (fileName, "r")) == NULL) {
    printf("The file %s could not be opened.\n", fileName);
    return 0;
  }
  printf("1\n");
  while (getc (dataFile) != ',');
  for (int i = 0; i < numSequences; i++) {

    // skip first column
    while (getc (dataFile) != ',');
    printf("%d\n", i);
    fscanf (dataFile, "%s", sequences[i]);
    *(sequences + (i + 1) * (sequenceLength + 1)) = '\0';

  }
  printf ("2\n");
  fclose (dataFile);
  return 1;
}

void printSequences (char ** sequences, int numSequences) {
  for (int i = 0; i < numSequences; i++)
    printf("sequence %d = %s\n", i, *(sequences + i));
}

int main (int argc, char *argv[]) {
  
  char fileName[100];
  strcpy (fileName, "../data/");
  int i = 0;
  int numSequences = 125;
  int sequenceLength = 201;
  char sequences[numSequences][sequenceLength + 1];
  // char ** sequences = (char **) malloc (numSequences * (sequenceLength) * sizeof (char));

  // append argv[1] to the end of path to data folder
  if (argc > 1) 
    while ((fileName[(i++) + 8] = argv[1][i]) != '\0');
  else {
    printf ("Please run again with a filename input from the data folder.\n");
    return 1;
  }

  // if (!readSequences (fileName, sequences, numSequences, sequenceLength))
  //  printf ("error reading data\n");


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
    //   **(sequences + i * sequenceLength) = '\0';
  }
  
  fclose (dataFile);

  // for (int i = 0; i < numSequences; i++)
  //  printf("sequence %d = %s\n", i, sequences[i]);
  // printSequences (sequences, numSequences);

  return 0;
}
