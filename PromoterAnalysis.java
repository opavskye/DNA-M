/*
 * Author:  Erik Opavsky
 * Date:	August, 2012
 * For:		Opavsky Lab, UNMC Cancer Research
 * 
 * This program looks over a set of methylation data
 * and extracts rows based on certain conditions.
 * 
 * The first condition is that the Distance_TSS column
 * is positive.
 * The second condition is that there are at least 10
 * unique Center_CCGG_HG_19 values for any geneSymbol.
 * However, not just the unique Center_CCGG_HG_19 values
 * are copied over to the output file, but the entire
 * geneSymbol group is copied over for the values under
 * which both the first condition holds true and the 
 * geneSymbol groups under which the second condition holds 
 * true.
 * 
 * In addition, when the values are copied over, there will
 * be a new column created titled "length" which stores the
 * value of txEnd - txStart.
 * 
 * ****************** IMPORTANT PROGRAM NOTE ******************
 * It must hold true that each data set is grouped by 
 * geneSymbol, or else this program will not function correctly.
 */

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

public class PromoterAnalysis {

	public static void main (String[] args) throws IOException {

		String outFile = args[0];

		int numInFiles = args.length - 1;
		String[] inFiles = new String[numInFiles];
		for (int i = 0; i < numInFiles; i++)
			inFiles[i] = args[i + 1];

		// look through header, find where txEnd, txStart, Distance_TSS, 
		HeaderInfo headInfo = analyzeHeader(inFiles[0], outFile);

		// check to make sure all went well in analyzeHeader
		if (headInfo.checkValid() > -1) {
			System.err.println ("ERROR:  could not find string:\t" + HeaderInfo.stringValue(headInfo.checkValid())
					+ "\tin file " + inFiles[0]);
			return;
		}

		// System.out.println (headInfo);
		// open output file for appending (since we already created it with the header)
		PrintWriter out = new PrintWriter (new BufferedWriter (new FileWriter (outFile, true)));


		int txStart = headInfo.getTxStartLoc();
		int txEnd = headInfo.getTxEndLoc(); 
		int distanceTss = headInfo.getDistanceTssLoc();
		int center19Loc = headInfo.getCenterCCGGHg19Loc();
		int geneSymbol = headInfo.getGeneSymbolLoc();
		int length = headInfo.getLengthLoc();


		// go through all of the input files
		for (String inFile : inFiles) {

			// open current file
			FileReader inputFile = new FileReader (inFile);
			Scanner fileIn = new Scanner (inputFile);

			// skip first line
			String line = fileIn.nextLine();

			ArrayList<String> lines = new ArrayList<String>();
			ArrayList<Integer> lengths = new ArrayList<Integer>();
			ArrayList<Double> uniqueHG19s = new ArrayList<Double>();
			String lastGeneSymbol = "";



			// go through every line of the file
			while (fileIn.hasNextLine()) {

				// read in next line from file
				line = fileIn.nextLine();
				String[] splitLine = line.split("\\s+");
				cleanLine (splitLine);

				// make sure this row is valid
				if (distanceTss > splitLine.length || 
						txEnd > splitLine.length || 
						txStart > splitLine.length ||
						geneSymbol > splitLine.length ||
						Double.parseDouble (splitLine[distanceTss]) < 0 ||
						Double.parseDouble (splitLine[center19Loc]) < 1000)
					continue;

				// if starting new group of geneSymbols
				if (!splitLine[geneSymbol].equals (lastGeneSymbol)) {

					// write all stuff to file if passes >= 10 unique HG19s requirement
					if (uniqueHG19s.size() >= 10) 
						for (int i = 0; i < lines.size(); i++) 
							out.write (lineWithLength (lines.get(i), lengths.get(i), length));	

					// reset everything to be ready for next geneSymbol group
					lines.clear();
					lengths.clear();
					uniqueHG19s.clear();
					lastGeneSymbol = splitLine[geneSymbol];
				}

				// add this line to the lines to be copied over if there are enough unique hg19s
				lines.add (line);
				lengths.add (Integer.parseInt (splitLine[txEnd]) - Integer.parseInt (splitLine[txStart]));

				// check to see if this HG19 is unique
				boolean foundHG19 = false;
				double currentHG19 = Double.parseDouble (splitLine[center19Loc]);
				for (Double HG19 : uniqueHG19s)
					if (currentHG19 == HG19)
						foundHG19 = true;

				if (!foundHG19)
					uniqueHG19s.add (currentHG19);

			} // next line now

			// close input file
			inputFile.close();
			fileIn.close();
		}

		// close output file
		out.close();

		System.out.println ("All " + numInFiles + " files have been successfully processed.");
	}

	private static String lineWithLength (String line, int length, int lengthLocation) {
		String s = new String();
		String[] splitLine = line.split("\\s+(?=([^\"]*\"[^\"]*\")*[^\"]*$)");

		for (int i = 0; i < splitLine.length + 1; i++) 
			if (i < lengthLocation)
				s += splitLine[i] + "\t";
			else if (i == lengthLocation)
				s += "\"" + length + "\"" + "\t";
			else if (i > lengthLocation)
				s += splitLine[i - 1] + "\t";

		s += "\n";

		return s;
	}

	private static void cleanLine (String[] data) {
		for (int i = 0; i < data.length; i++) 
			data[i] = data[i].replace ("\"", "");
	}

	private static HeaderInfo analyzeHeader (String inFile, String outFile) {

		HeaderInfo headInfo = new HeaderInfo (-1, -1, -1, -1, -1, -1);

		try
		{
			FileReader inputFile = new FileReader (inFile);
			Scanner fileIn = new Scanner (inputFile);

			String header = fileIn.nextLine();

			String[] head = header.split("\\s+(?=([^\"]*\"[^\"]*\")*[^\"]*$)");

			int txStart = -1;
			int txEnd = -1; 
			int distanceTss = -1;
			int center19Loc = -1;
			int geneSymbol = -1;
			int length = -1;

			for (int i = 0; i < head.length; i++) {
				if (head[i].equals ("\"Center_CCGG_hg19\""))
					center19Loc = i;
				else if (head[i].equals ("\"txEnd\""))
					txEnd = i;
				else if (head[i].equals ("\"txStart\""))
					txStart = i;
				else if (head[i].equals ("\"Distance_TSS\""))
					distanceTss = i;
				else if (head[i].equals ("\"geneSymbol\"")) {
					geneSymbol = i;
					length = i + 1;
				}				
			}

			headInfo = new HeaderInfo (txStart, txEnd, distanceTss, center19Loc, geneSymbol, length);
			// System.out.println (headInfo);

			inputFile.close();
			fileIn.close();

			PrintWriter out = new PrintWriter (new BufferedWriter (new FileWriter (outFile)));

			for (int i = 0; i < head.length + 1; i++) {
				if (i < length)
					out.write (head[i] + "\t");
				else if (i == length)
					out.write ("\"length\"" + "\t");
				else if (i > length)
					out.write (head[i - 1] + "\t");
			}

			out.write('\n');
			out.close();

		} 
		catch (IOException e)
		{
			System.err.println(e);
		}

		return headInfo;
	}
}
