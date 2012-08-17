#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <float.h>

int isSpaceOrTab (char c) {
  return (c == ' ' || c == '\t');
}

int isWhiteOrQuote (char c) {
  return (isspace (c) || c == '"');
}

int analyzeHeader (char * outFileName, char * inFileName, char * columnString) {
  FILE * inFile;
  FILE * outFile;
  int distance = -1;
  int maxLineSize = 10000;
  int maxColumns = 1000;
  char columnFinder[maxLineSize];

  // open inFile
  if (!(inFile = fopen (inFileName, "r"))) {
    printf ("Error opening file %s.  Aborting.\n", inFileName);
    return -1;
  }

  // find which column Distance_TSS is in
  int i;
  for (i = 0; i < maxColumns; i++) {
    fscanf (inFile, "%s", columnFinder);
    if (!strcmp (columnFinder, columnString))
      distance = i;
  }

  if (distance == -1) {
    printf ("ERROR:  Could not find %s in file %s.  Aborting.\n", columnString, inFileName);
    fclose (inFile);
    return -1;
  } 
  
  // reopen the file to start again at the beginning
  fclose (inFile);
  if (!(inFile = fopen (inFileName, "r"))) {
    printf ("Error opening file %s.  Aborting.\n", inFileName);
    return -1;
  }
  
  // open outFile to put in the header
  if (!(outFile = fopen (outFileName, "w"))) {
    printf ("Error opening file %s.  Aborting.\n", outFileName);
    return -1;
  }

  // transfer the header (first row)
  fgets (columnFinder, maxLineSize, inFile);
  fprintf (outFile, "%s", columnFinder);

  // close both files
  fclose (inFile);
  fclose (outFile);

  return distance;
}

double getFromColumn (char * source, int columnNum) {
  
  char c;
  int i, j = 0;

  // to store the value in the query column
  char val[50];

  for (i = 0; i < columnNum; i++) {

    // go through ith column
    while (!isSpaceOrTab (c = source[j++]))
      if (c == '\n')
        return DBL_MAX;

    // go through ith whitespace
    while (isSpaceOrTab (c = source[j++]))
      if (c == '\n')
        return DBL_MAX;
  }

  i = 0;
  while (!isWhiteOrQuote (c = source[j]))
    val[i++] = source[j++];

  val[i] = '\0';

  return atof (val);
}

int main (int argc, char *argv[]) {

  if (argc < 5) {
    printf ("Error:  correct program usage is ./extractor minBound maxBound outputFile inputFile1 inputFile2 ...\n");
    return 1;
  }

  double minBound = atof (argv[1]);
  double maxBound = atof (argv[2]);
  char * outFileName = argv[3];
  FILE * outFile;
  int numFiles = argc - 4;
  char ** inFile = argv + 4;
  FILE * currentFile;
  char temp;
  char * columnString = "\"Distance_TSS\"";

  int maxRowLength = 20000;
  char row[maxRowLength];
  double value;

  // find which column the columnString is in and transfer the header from inFile[0] to outFile
  int distanceColumn = analyzeHeader(outFileName, inFile[0], columnString);

  if (distanceColumn == -1)
    return 1;

  // open output file
  if (!(outFile = fopen (outFileName, "a"))) {
    printf ("Error opening file %s.  Aborting.\n", outFileName);
    return 1;
  }

  // go through all files
  int i;
  for (i = 0; i < numFiles; i++) {

    // open file
    if (!(currentFile = fopen (inFile[i], "r"))) {
      printf ("Error opening file %s.  Aborting.\n", inFile[i]);
      return 1;
    }
       
    // skip first header row of the current file
    while ((temp = fgetc (currentFile)) != '\n');

    // go through all rows, checking them one at a time
    while (fgets (row, maxRowLength, currentFile) != NULL) {
      
      // get query column value
      value = getFromColumn (row, distanceColumn);

      // check bounds and copy row over if value fits
      if (value >= minBound && value <= maxBound)
        fprintf (outFile, "%s", row);

    }

    fclose (currentFile);
  }

  fclose (outFile);

  // check if we've gone through all of the files
  if (i == numFiles)
    printf ("All %d files have been successfully processed.\n", numFiles);
  else
    printf ("Error processing files.\n");

  return 0;
}
