package ipredict.predictor.CPT.CPTPlus;

import java.util.HashMap;
import java.util.List;

import ipredict.database.Item;
import ipredict.database.Sequence;

public interface FIF {

	public HashMap<Item, Integer> getItemFrequencies(List<Sequence> seqs);
	
	public List<List<Item>> findFrequentItemsets(List<Sequence> seqs, int minLength, int maxlength, int minSup);
}
