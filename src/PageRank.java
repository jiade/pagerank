import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.Counter;


public class PageRank {
	
	static enum CountersEnum { NUMBER_NODES, SUM_CONVERGENCE }
	
	// the input data are reformatted as [nodeId, pagerank, outlink list]
	// Map input to pair (node_id, outlink list and pagerank term)
	public static class PageRankMapper
	extends Mapper<Object, Text, Text, Text>
	{

		public void map(Object key, Text value, Context context
				) throws IOException, InterruptedException 
		{
			Text k = new Text();
			Text v = new Text(); // (k, v) used to write to context
			
			String[] ids = value.toString().split("\\s+");
			String k_str=null, v_str=null;
			if (ids.length >= 2) // make sure the input format is correct
			{
				// calculate pagerank term to be passed to reducer
				double pagerank = Double.parseDouble(ids[1]);
				// represent out link list string
				String out_link_list = new String("|");
				for (int i=2; i < ids.length; i++)
				{
					double pagerank_term = pagerank / (ids.length - 2);
					out_link_list += ("->" + ids[i]); // construct the list
					k_str = ids[i];
					v_str = String.valueOf(pagerank_term);
					k.set(k_str);
					v.set(v_str);
					System.out.println("Map to("+k_str+", "+v_str+")");
					context.write(k, v);
				}
				// pass the link list to reducer as well
				k_str = ids[0];
				v_str = out_link_list;
				k.set(k_str);
				v.set(v_str);
				System.out.println("Map to("+k_str+", "+v_str+")");
				context.write(k, v);
			}

		}
	}

	public static class PageRankReducer
	extends Reducer<Text,Text,Text,Text> {
		

		public void reduce(Text key, Iterable<Text> values,
				Context context
				) throws IOException, InterruptedException 
		{
			//long N = node_num;
			double sum = 0;
			String out_link_list = null;
			for (Text val : values) 
			{
				String val_str = val.toString();
				if (val_str.startsWith("|")) // is link list
				{
					// reconstruct the link list as output format
					out_link_list = val_str.substring(1).replaceAll(Pattern.quote("->"), " ");
				}
				else
				{
					sum += Double.parseDouble(val.toString());
				}
			}
			Configuration conf = context.getConfiguration();
			long N = Long.parseLong(conf.get("number of nodes"));
			double DAMPING_FACTOR = Double.parseDouble(conf.get("damping factor"));
			double page_rank = (1 - DAMPING_FACTOR)/N + (DAMPING_FACTOR * sum);
			String k_str = key.toString();
			String v_str = String.valueOf(page_rank) + out_link_list;
			System.out.println("Reduce to output("+k_str+", "+v_str+")");
			context.write(key, new Text(v_str));
		}
	}


	// use this mapper class (and reducer) to calculate how many nodes in the file,
	// how many edges, max/min out degree, etc.
	public static class GraphPropertyMapper
	extends Mapper<Object, Text, Text, Text>
	{
		@Override
		public void map(Object key, Text value, Context context
				) throws IOException, InterruptedException 
		{
			String[] ids = value.toString().split("\\s+");
			if (ids[0].equals("") && ids.length == 1)
			{
				return; // empty line, just ignore
			}
			// count how many nodes in the file since each map indicates one line of text
			Counter counter = context.getCounter(CountersEnum.class.getName(),
		            CountersEnum.NUMBER_NODES.toString());
			counter.increment(1);
			// count how many edges for current node, output [nid:nedge] as value
			if (ids.length - 2 > 0 && ids[2].equals(""))
			{
				// deal with trailing space problem if there is any
				context.write(new Text("graph"), new Text(ids[0]+":0"));
			}
			else
			{
				context.write(new Text("graph"), new Text(ids[0]+":"+ String.valueOf(ids.length - 2)));
			}
		}
	}
	public static class GraphPropertyReducer
	extends Reducer<Text,Text, Text, Text> 
	{
		public void reduce(Text key, Iterable<Text> values,
				Context context
				) throws IOException, InterruptedException 
		{
			// calculates min/max/avg edges problem.
			int sumEdges = 0, num = 0;
			int maxEdge = -1;
			String maxEdgeNodeId = null, minEdgeNodeId = null;
			int minEdge = Integer.MAX_VALUE;
			String output_nid_edge = new String();
			for (Text val : values)
			{
				String val_str = val.toString();
				String nid = val_str.split(":")[0];
				String edge = val_str.split(":")[1];
				int nedge = Integer.parseInt(edge);
				output_nid_edge += String.format("node [%s] has out-degree = %d\n", nid, nedge);
				sumEdges += nedge;
				// find min and max value of degree
				if (nedge > maxEdge)
				{
					maxEdge = nedge;
					maxEdgeNodeId = nid;
				}
				if (nedge < minEdge)
				{
					minEdge = nedge;
					minEdgeNodeId = nid;
				}
				num++;
			}
			
			float avgEdges = 0;
			if (num != 0) 
			{
				avgEdges = (float) sumEdges / num;
			}
			// organize output
			String output_str = String.format("\n\nnumber of nodes = %d\n", num);
			output_str += String.format("number of edges (directional) = %d\n", sumEdges);
			output_str += String.format("node [%s] has min out-degree = %d\n", minEdgeNodeId, minEdge);
			output_str += String.format("node [%s] has max out-degree = %d\n", maxEdgeNodeId, maxEdge);
			output_str += String.format("average of out-degree = %f\n\n", avgEdges);
			
			output_str += "Following are details of each node's out-degrees:\n\n";
			output_str += output_nid_edge;

			context.write(new Text("Graph Property Summary:"), new Text(output_str));
		}
	}
	// use this mapper class to reformat input file as the following code requires.
	public static class ReformatMapper
	extends Mapper<Object, Text, Text, NullWritable>
	{
		@Override
		public void map(Object key, Text value, Context context
				) throws IOException, InterruptedException 
		{
			String[] ids = value.toString().split("\\s+");
			if (ids.length == 0 || (ids.length == 1 && ids[0].equals("")))
			{
				return; // empty line, just ignore
			}
			String k_strs[] = value.toString().split("\\s+", 2);
			Text k;
			Configuration conf = context.getConfiguration();
			double INIT_PR = Double.parseDouble(conf.get("initial pagerank value"));
			if (k_strs.length > 1) // always be aware of space problems
			{
				k = new Text(k_strs[0] + " " + String.valueOf(INIT_PR) + " "
						+ k_strs[1]);
			}
			else
			{
				k = new Text(k_strs[0] + " " + String.valueOf(INIT_PR));				
			}
			context.write(k, NullWritable.get());
		}
	}

	// use this mapper class to make descending ranking for results' page rank value
	public static class RankingMapper
	extends Mapper<Object, Text, Text, NullWritable>
	{
		@Override
		public void map(Object key, Text value, Context context
				) throws IOException, InterruptedException 
		{
			String[] ids = value.toString().split("\\s+");
			if (ids.length >= 2) // make sure the line format is correct
			{
				String k_str = ids[0] + "\t" + ids[1];
				context.write(new Text(k_str), NullWritable.get());
			}
		}
	}
	// helper class to do descending sorting on keys
	public static class DescendingFloatComparator extends WritableComparator {
	    protected DescendingFloatComparator() {
	        super(Text.class, true);
	    }

	    @SuppressWarnings("rawtypes")
	    @Override
	    public int compare(WritableComparable w1, WritableComparable w2) {
	        Text key1 = (Text) w1;
	        Text key2 = (Text) w2;      
	        FloatWritable n1 = new FloatWritable(Float.valueOf(key1.toString().split("\\s+")[1]));
	        FloatWritable n2 = new FloatWritable(Float.valueOf(key2.toString().split("\\s+")[1]));
	        
	        return -1 * n1.compareTo(n2);
	    }
	}

	// this mapper/reducer is used for calculating convergence condition for page rank iterations
	public static class ConvergenceMapper
	extends Mapper<Object, Text, Text, DoubleWritable>
	{
		@Override
		public void map(Object key, Text value, Context context
				) throws IOException, InterruptedException 
		{
			String[] ids = value.toString().split("\\s+");
			if (ids[0].equals("") && ids.length == 1)
			{
				return; // empty line, just ignore
			}
			if (ids.length >= 2) // make sure the input format is correct
			{
				Text k = new Text(ids[0]);
				DoubleWritable v = new DoubleWritable(Double.parseDouble(ids[1]));
				context.write(k, v);
			}

		}
	}
	public static class ConvergenceReducer
	extends Reducer<Text,DoubleWritable, NullWritable, NullWritable> 
	{
		public void reduce(Text key, Iterable<DoubleWritable> values,
				Context context
				) throws IOException, InterruptedException 
		{
			double pagerank[] = new double[2];
			int i = 0; // can't be greater than 2 elements
			for (DoubleWritable val : values)
			{
				pagerank[i] = val.get();
				++i;
			}
			double diff = Math.abs(pagerank[0] - pagerank[1]);
			// use counter to store the shared data
			Counter counter = context.getCounter(CountersEnum.SUM_CONVERGENCE);
			Configuration conf = context.getConfiguration();
			double CONVG_PRECISION = Double.parseDouble(conf.get("convergence threshold"));
			long diff_counter = (long)(diff / CONVG_PRECISION);
			//long sum = counter.getValue();
			//counter.setValue(sum + diff_counter);
			counter.increment(diff_counter);
		}
	}

	
	public static void main(String[] args) throws Exception 
	{
		final double DAMPING_FACTOR = 0.85; 
		final double INIT_PR = 1.0;
		final double CONVG_PRECISION = 1.0E-3;

		int constIter = 0;
		if (args.length > 2) // third argument indicates we use constant iteration runs
		{
			constIter = Integer.parseInt(args[2]);
		}
		Path inputPath = new Path(args[0]);
		Path outputPath = new Path(args[1]);
		Path workingDir = inputPath.getParent();
		Path tempPath = new Path(workingDir, "temp");
		
		
		//-------------- STEP 1: preprocessing... reformat input file format
		// output to temp/input0 folder in which each column represents [node id; page rank value; link list]
		// the temp/input0 folder will be used as new input for following operations.
		Configuration formatterConf = new Configuration();
		formatterConf.set("initial pagerank value", String.valueOf(INIT_PR));
		Job formatterJob = new Job(formatterConf, "reformatter hadoop-0.20");
		formatterJob.setJarByClass(PageRank.class);
		formatterJob.setMapperClass(ReformatMapper.class);
		formatterJob.setOutputKeyClass(Text.class);
		formatterJob.setOutputValueClass(NullWritable.class);
		FileInputFormat.addInputPath(formatterJob, inputPath);
		FileOutputFormat.setOutputPath(formatterJob, new Path(tempPath, "input0"));
		formatterJob.waitForCompletion(true);
		
		
		//-------------- STEP 2: preprocessing... count how many nodes in the file and also calculates graph property
		Configuration gPropertyConf = new Configuration();
		Job gPropertyJob = new Job(gPropertyConf, "get graph property hadoop-0.20");
		gPropertyJob.setJarByClass(PageRank.class);
		gPropertyJob.setMapperClass(GraphPropertyMapper.class);
		gPropertyJob.setReducerClass(GraphPropertyReducer.class);
		//numNodeJob.setOutputFormatClass(NullOutputFormat.class);
		gPropertyJob.setOutputKeyClass(Text.class);
		gPropertyJob.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(gPropertyJob, new Path(tempPath, "input0"));
		FileOutputFormat.setOutputPath(gPropertyJob, new Path(outputPath, "Graph_property"));
		gPropertyJob.waitForCompletion(true);
		long N = gPropertyJob.getCounters().findCounter(CountersEnum.NUMBER_NODES).getValue();
		System.out.println("N = "+ N);
		
		
		//-------------- STEP 3: calculate page rank iteratively
		// the input are in temp/input[run] folder
		int run = 0;
		long delta = Long.MAX_VALUE;
		long start_time = System.currentTimeMillis(); // time the job
		long run_10_time = start_time;
		for (; constIter > 0 ? run < constIter : delta > 0;)
		{
			Configuration conf = new Configuration();
			conf.set("number of nodes", String.valueOf(N));
			conf.set("damping factor", String.valueOf(DAMPING_FACTOR));
			Job job = new Job(conf, "Page Rank hadoop-0.20");
			job.setJarByClass(PageRank.class);
			job.setMapperClass(PageRankMapper.class);
			job.setReducerClass(PageRankReducer.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);
			FileInputFormat.addInputPath(job, new Path(tempPath, "input"+ String.valueOf(run)));
			FileOutputFormat.setOutputPath(job, new Path(tempPath,"input" + String.valueOf(run+1)));
			job.waitForCompletion(true);
			
			//---------- STEP 3+: calculate if convergence condition is met
			// start another map-reduce job to calculate page rank value change between iterations,
			// this value will be used for convergence criterion determination.
			Configuration convConf = new Configuration();
			convConf.set("convergence threshold", String.valueOf(CONVG_PRECISION));
			Job convJob = new Job(convConf, "sum convergence hadoop-0.20");
			convJob.setJarByClass(PageRank.class);
			convJob.setMapperClass(ConvergenceMapper.class);
			convJob.setReducerClass(ConvergenceReducer.class);
			convJob.setMapOutputKeyClass(Text.class);
			convJob.setMapOutputValueClass(DoubleWritable.class);
			convJob.setOutputKeyClass(NullWritable.class);
			convJob.setOutputValueClass(NullWritable.class);
			FileInputFormat.addInputPath(convJob, new Path(tempPath, "input"+ String.valueOf(run)));
			FileInputFormat.addInputPath(convJob, new Path(tempPath, "input"+ String.valueOf(run+1)));
			FileOutputFormat.setOutputPath(convJob, new Path(
					tempPath, "delta_" + String.valueOf(run)+"-"+String.valueOf(run+1)));
			//convJob.getCounters().findCounter(CountersEnum.SUM_CONVERGENCE).setValue(0);
			convJob.waitForCompletion(true);
			// retrieve current pagerank diff
			delta = convJob.getCounters().findCounter(CountersEnum.SUM_CONVERGENCE).getValue();
			System.out.println("current delta = "+ delta);
			run++;
			if (run == 10)
			{
				run_10_time = System.currentTimeMillis();
			}
		}
		long end_time = System.currentTimeMillis(); // end the timer
		float wholeDuration = (float)(end_time - start_time)/1000; // in seconds
		float time10Duration = (float)(run_10_time - start_time)/1000; // in seconds
	
		//-------------- STEP 4: ranking the result in descending order and arrange output with clean-up
		Configuration rankingConf = new Configuration();
		Job rankingJob = new Job(rankingConf, "ranking job hadoop-0.20");
		rankingJob.setJarByClass(PageRank.class);
		rankingJob.setMapperClass(RankingMapper.class);
		rankingJob.setOutputKeyClass(Text.class);
		rankingJob.setOutputValueClass(NullWritable.class);
		rankingJob.setSortComparatorClass(DescendingFloatComparator.class);
		FileInputFormat.addInputPath(rankingJob, new Path(tempPath, "input"+ String.valueOf(run)));
		Path outputRankedPath = new Path(outputPath, "results_ranked");
		FileOutputFormat.setOutputPath(rankingJob, outputRankedPath);
		rankingJob.waitForCompletion(true);
		// clean up results to targeted output
		try 
		{
			FileSystem hdfs = FileSystem.get(new Configuration());
			// extract top 10 lines from results and write to the statistics file
			Path mergedFile = new Path(tempPath, "merged.txt");
			FileUtil.copyMerge(hdfs, outputRankedPath, hdfs, mergedFile, false, new Configuration(), "");
			BufferedReader br=new BufferedReader(new InputStreamReader(hdfs.open(mergedFile)));
			Path pt = new Path(outputPath, "statistics.txt");
			BufferedWriter bw=new BufferedWriter(new OutputStreamWriter(hdfs.create(pt,true)));
			// write statistics
			bw.write(String.format("number of iterations = %d\n", run));
			bw.write(String.format("execution time for first 10 runs = %.3f seconds\n", time10Duration));
			bw.write(String.format("total execution time = %.3f seconds\n\n", wholeDuration));
			bw.write("Top 10 largest page rank nodes are: \n");
			String str = null;
			for (int cnt = 0; cnt < 10;)
			{
				if ((str = br.readLine()).equals(""))
				{
					continue; // just ignore empty line (in case)
				}
				bw.write(str +"\n");
				cnt++;
			}
			bw.close();
			// clean up temp folders
			//hdfs.delete(tempPath, true);
		} 
		catch (Exception e) 
		{
			// Auto-generated catch block
			e.printStackTrace();
		}
	}

}
