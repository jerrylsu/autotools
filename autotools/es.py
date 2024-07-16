"""ElasticSearch tools.
"""
import json
import elasticsearch
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm
from typing import List, Dict, Optional, NamedTuple, Iterator
from loguru import logger


class BatchedSearchResults(NamedTuple):
    total_scores: List[List[float]] = None
    total_indices: List[List[int]] = None


class ElasticSearchIndex(object):
    """
    Sparse index using Elasticsearch. It is used to index text and run queries based on BM25 similarity.
    An Elasticsearch server needs to be accessible, and a python client is declared with
    ```
    es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    ```
    """
    def __init__(self, host: Optional[str] = "localhost", port: Optional[int] = 9200):
        self.es_client = elasticsearch.Elasticsearch(hosts=[{"host": host, "port": port, "scheme": "http"}],
                                                     # http_auth=(es_user, es_passwd),
                                                     max_retries=3,
                                                     request_timeout=120,
                                                     retry_on_timeout=True)
        logger.info("ElasticSearch connected {}:{} successfully.".format(host, str(port)))

    def create_index(self, index_name: str, index_file: str) -> str:
        """Create elasticsearch index.
        https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html

        Args:
            index_name (`str`): The index name.
            index_file (`Dict`): The index mapping file.
        Output:
            status (`str`): The status, success "OK", fail "ERROR".
        """
        status = "OK"
        if self.es_client.indices.exists(index=index_name):
            logger.info(f"ES index [{index_name}] already exists.")
            return status

        with open(index_file, "r") as index_file:
            source = json.load(index_file)

        try:
            ret = self.es_client.indices.create(index=index_name, body=source)
            if not ret["acknowledged"]:
                logger.error(f"ES index create failed: {ret}")
                status = "ERROR"
            logger.info(f"Create ES index: {index_name}.")
        except Exception as err:
            logger.error(f"ES index create failed: {err}")
            status = "ERROR"
        return status

    def delete_index(self, index_name: str):
        """Delete elasticsearch index.

        Args:
            index_name (`str`): The index name.
        Output:
            status (`str`): The status, success "OK", fail "ERROR".
        """ 
        status = "OK"
        if not self.es_client.indices.exists(index=index_name):
            logger.info(f"Not exists ES index: {index_name}.")
            return status

        try:
            ret = self.es_client.indices.delete(index=index_name)
            if not ret["acknowledged"]:
                logger.error(f"ES index delete failed: {ret}")
                status = "ERROR"
            logger.info(f"Delete ES index: {index_name}.")
        except Exception as err:
            logger.error(f"ES index delete failed: {err}")
            status = "ERROR"
        return status

    def add_documents(self, index_name: str, docs_generator: Iterator) -> str:
        """Add documents to the elasticsearch index.

        Args:
            index_name (`str`): The index name.
            docs_generator (`Iterator`): The documents generator.
        Output:
            status (`str`): The status, success "OK", fail "ERROR".
        """
        status = "OK"
        try:
            successes = 0
            progress = tqdm(unit="docs")
            for ok, action in streaming_bulk(client=self.es_client, index=index_name, actions=docs_generator):
                progress.update(1)
                successes += ok
        except Exception as err:
            logger.error(f"ES add documents failed: {err}")
            status = "ERROR"

        logger.info(f"Add documents to index [{index_name}]: {successes}.")
        return status

    def search(self, index_name: str, query: Dict) -> Dict:
        """Search query.

        Args:
            index_name (`str`): The index name.
            query (`Dict`): The query statement.
        Output:
            response (`Dict`): The response of elasticsearch.
        """
        try:
            search_start = time.time()
            response = self.es_client.search(index=index_name, body=query)
            search_time = time.time() - search_start
            logger.info(f"Search time: {search_time}")
        except Exception as err:
            logger.error(f"Search the nearest documents indices {index_name} error: {err}.")
            return {}
        return response

    def search_batch(self, queries: List[str], size: int = 5, max_workers: int = 10) -> BatchedSearchResults:
        """Find the nearest documents indices to the batch query.

        Args:
            queries (`str`): The query as a string.
            size (`int`): The number of documents to retrieve.
            max_workers (`int`): The number of thread.
        Output:
            scores (`List[List[float]`): The retrieval scores of the retrieved documents.
            indices (`List[List[int]]`): The indices of the retrieved documents.
        """
        import concurrent.futures

        total_scores, total_indices = [None] * len(queries), [None] * len(queries)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.search, query, size): i for i, query in enumerate(queries)}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                results: SearchResults = future.result()
                total_scores[index] = results.scores
                total_indices[index] = results.indices
        return BatchedSearchResults(total_scores=total_scores, total_indices=total_indices)

