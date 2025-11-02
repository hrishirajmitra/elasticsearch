# Indexing and retrieval assignment
The goal is to build a search index on data and understand some aspects of the internals. In the first part you will simply use Elasticsearch to index the data and enable querying. The second part will involve implementing an indexing and querying system from scrarch. We will list some resources and give you some boilerplate code to begin with. You are free to choose exact details of how you would like implement it like specific datastore, choice of algorithm, etc. We will begin with one data source and expand to other domains in later parts. The assignment will be released in small parts (activities) each of which should not take more than 3-5 hours for a beginner and with consistent progress you should get done without much trouble. Please feel free to reach out for any help.
Make sure you add all your code as you work on it to a github repository. We should be able to clone your repo and run it for evaluation. When the github classroom setup is ready you will need to put your code in the given structure (it will be python boilerplate code, let us know if you cannot use python) and put it there for grading. You will update the repo as further assignment activities are released below through rest of this course. So we are refraining from adding any specific due dates. However, we will have multiple  automated/manual checkpoints for evaluation that you will need to meet. For each of the activities below, the following artefacts should be generated as indicated along each of the items.
   + `A`: System response time (latency) across a query set with p95 and p99 (95th and 99th percentiles). Create a set that is diverse enough by probing an LLM and justify how it captures various required system properties.
   + `B`: System throughput in queries/second for read or write operations or a mix of both as appropriate for the activity.
   + `C`: Memory footprint of the system.
   + `D`: Functional metrics like precision, recall or ranking measures.

## Activity
1. Data sources for experiments: preprocess the data including word stemming and stopping, handle any special symbols like punctuation appropriately. Generate word frequency plots with and without text preprocessing. Index the data into Elasticsearch, name it `ESIndex-v1.0`.
   1. News data at [webz.io](https://github.com/Webhose/free-news-datasets) available on github.
   2. Wiki data from [huggingface](https://huggingface.co/datasets/wikimedia/wikipedia) use the split `20231101.en`.
1. Implement your own simple indexing (`SelfIndex-v1.0`) over the boilerplate code shared and index the above data into it.
   1. Build `SelfIndex-v1.xyziq` identified by the versioning number as follows. Structure your code so that you can use this version number to have it build/make the index with right choices and run the query tests on it accordingly. Plot relevant metrics (marked below) for your indexing against that of ES for the corresponding test query set.
      1. `Plot.C` for `x=n` Iterate over different kinds of information indexed.
         1. `x=1` Boolean index with document IDs and postition IDs.
         2. `x=2` Enable ranking with word counts.
         3. `x=3` Further evaluate gains from adding TF-IDF scores.
      1. `Plot.A` for `y=n` Compare different datastore choices.
         1. `y=1` uses custom objects (using Python's pickling, JSONs or other such methods) stored on local disk
         1. `y=2` uses any two off-the-shelf choices (like PostgresSQL GIN, RocksDB, Redis, etc). You will need to understand, at a very high-level, what the DB implements and discuss its pro and cons.
      1. `Plot.AB` for `z=n` compares two compression methods on the postings list, one simple code (`z=1`) and an off-the-shelf library (`z=2`).
      1. `Plot.A` for `i=0/1` implements index optimization like skipping with pointers.
      1. `Plot.AC` for `q=Tn/Dn` implements query processing engine with Term-at-a-time (`q=Tn`) and Document-at-a-time (`q=Dn`) with `n` indicating any optimizatin implemented.

## Resources
1. https://elasticsearch-py.readthedocs.io/en/v9.0.3/
1. https://elasticsearch-dsl.readthedocs.io/en/latest/
1. https://www.elastic.co/docs/reference/elasticsearch/clients/python
1. https://last9.io/blog/elasticsearch-python-integration/


## Additional context
Indexing is the process of organizing and structuring data to enable fast and efficient retrieval of information. In the context of information retrieval, an index allows you to quickly find documents that contain specific terms or satisfy certain queries, without scanning every document in the collection. For this assignment, you will build an index over some texts. The process involves:

1. _Extracting terms_: For given text, read the text, tokenize it, remove stop word and perform stemming over resulting tokens to obtain terms.
2. _Building an index (Boolean index)_
- Build an inverted index (also known as a boolean index) that maps each term to the list of documents in which it appears.
- This index should support boolean queries using the operators `AND`, `OR`, `NOT`, and `PHRASE` operators, and allow the use of parentheses for grouping.
	- Example query: `("Apple" AND "Banana") OR ("Orange" AND NOT "Grape")`
	- Operator Precedence: Order of precedence from highest to lowest is `PHRASE`, `NOT` `AND`, and `OR`.
    - Query Grammar
  
            
            QUERY    := EXPR
            EXPR     := TERM | (EXPR) | EXPR AND EXPR | EXPR OR EXPR | NOT EXPR | PHRASE EXPR
            TERM     := a single word/term surrounded with double quotes
			

3. _Persistence Requirement_
- Your index must be persisted on disk (e.g., using files, databases, or serialization). This ensures that the index is not lost if the server stops.
- When the server is started, your code should automatically load all previously created indices from disk, so that all indices are available without needing to rebuild them.
