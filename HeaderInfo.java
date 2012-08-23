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

public class HeaderInfo {

	private int txStartLoc;
	private int txEndLoc;
	private int distanceTssLoc;
	private int centerCCGGHg19Loc;
	private int geneSymbolLoc;
	private int lengthLoc;
	
	public HeaderInfo (int txStart, int txEnd, int distanceTss, int center19Loc, int geneSymbol, int length) {
		txStartLoc = txStart;
		txEndLoc = txEnd;
		distanceTssLoc = distanceTss;
		centerCCGGHg19Loc = center19Loc;
		geneSymbolLoc = geneSymbol;
		lengthLoc = length;
	}

	public int checkValid() {
		if (txStartLoc < 0)
			return 0;
		if (txEndLoc < 0) 
			return 1;
		if (distanceTssLoc < 0)
			return 2;
		if (centerCCGGHg19Loc < 0)
			return 3;
		if (geneSymbolLoc < 0)
			return 4; 
		if (lengthLoc < 0)
			return 5;

		return -1;
	}
	
	public static String stringValue (int index) {
		if (index == 0)
			return "\"txStart\"";
		else if (index == 1)
			return "\"txEnd\"";
		else if (index == 2)
			return "\"Distance_TSS\"";
		else if (index == 3)
			return "\"Center_CCGG_hg19\"";
		else if (index == 4)
			return "\"geneSymbol\"";
		else if (index == 5)
			return "\"geneSymbol\""; // that was intentional geneSymbol
		return "";
	}
	
	public int getTxStartLoc() {
		return txStartLoc;
	}

	public int getTxEndLoc() {
		return txEndLoc;
	}

	public int getDistanceTssLoc() {
		return distanceTssLoc;
	}

	public int getCenterCCGGHg19Loc() {
		return centerCCGGHg19Loc;
	}

	public int getGeneSymbolLoc() {
		return geneSymbolLoc;
	}
	
	public int getLengthLoc() {
		return lengthLoc;
	}

	@Override
	public String toString() {
		return "HeaderInfo [txStartLoc=" + txStartLoc + ", txEndLoc="
				+ txEndLoc + ", distanceTssLoc=" + distanceTssLoc
				+ ", centerCCGGHg19Loc=" + centerCCGGHg19Loc
				+ ", geneSymbolLoc=" + geneSymbolLoc + ", lengthLoc="
				+ lengthLoc + "]";
	}
}
