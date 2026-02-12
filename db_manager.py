# Neo4j Imports
from neo4j import GraphDatabase

class Neo4jManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def check_connection(self):
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection Successful' AS msg")
                return result.single()["msg"]
        except Exception as e:
            return f"Connection Failed: {e}"

    def setup_constraints(self):
        """Initializes all unique constraints for the Graph."""
        with self.driver.session() as session:
            # Content Hash: The ultimate ID for a document's content
            session.run("CREATE CONSTRAINT doc_hash IF NOT EXISTS FOR (d:Document) REQUIRE d.hash IS UNIQUE")
            # File Path: Prevents the same path being indexed twice
            session.run("CREATE CONSTRAINT doc_path IF NOT EXISTS FOR (d:Document) REQUIRE d.path IS UNIQUE")
            # Chunk ID: Unique identifier for text segments
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            print("All Constraints (Hash, Path, Chunk) initialized.")
        
    def get_processed_hashes(self):
        """Returns a set of all content hashes already in the DB."""
        with self.driver.session() as session:
            result = session.run("MATCH (d:Document) RETURN d.hash AS hash")
            return {record["hash"] for record in result}
        
    def is_hash_present(self, file_hash):
        """
        Fast Boolean check if a Document with this hash exists.
        Used to prevent re-embedding the same file from different sources.
        """
        with self.driver.session() as session:
            # We match strictly on the hash property
            query = "MATCH (d:Document {hash: $h}) RETURN count(d) > 0 as exists"
            result = session.run(query, h=file_hash).single()
            return result["exists"] if result else False
        
    def clear_data(self):
        """
        Deletes all nodes and relationships. 
        Note: This keeps your constraints and indexes intact.
        """
        with self.driver.session() as session:
            # DETACH DELETE removes the node AND any relationships connected to it
            result = session.run("MATCH (n) DETACH DELETE n")
            summary = result.consume()
            print(f"Data Cleared: Deleted {summary.counters.nodes_deleted} nodes "
                  f"and {summary.counters.relationships_deleted} relationships.")

    def clear_schema(self):
        """
        Removes all constraints and indexes from the database.
        """
        with self.driver.session() as session:
            # 1. Get all constraints
            constraints = session.run("SHOW CONSTRAINTS")
            for record in constraints:
                session.run(f"DROP CONSTRAINT {record['name']}")
            
            # 2. Get all indexes
            indexes = session.run("SHOW INDEXES")
            for record in indexes:
                # We skip lookup indexes which are system-managed
                if record['type'] != 'LOOKUP':
                    session.run(f"DROP INDEX {record['name']}")
                    
            print("Schema Cleared: All constraints and indexes removed.")

    def get_all_embedded_urls(self):
        """
        Returns a list of all URLs currently stored as WebPage nodes.
        """
        with self.driver.session() as session:
            result = session.run("MATCH (p:WebPage) RETURN p.url AS url ORDER BY p.url")
            urls = [record["url"] for record in result]
            print(f"Found {len(urls)} embedded web pages.")
            return urls

    def delete_domain_data(self, domain):
        """
        Deletes ALL WebPage nodes and their Chunks for a specific domain.
        Example: domain="docs.python.org" removes all pages from that site.
        """
        print(f"Deleting all data for domain: {domain}...")
        with self.driver.session() as session:
            # We match by the 'domain' property we will store on the WebPage node
            query = """
            MATCH (p:WebPage {domain: $domain})
            OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk)
            DETACH DELETE p, c
            """
            result = session.run(query, domain=domain)
            summary = result.consume()
            print(f"Deleted {summary.counters.nodes_deleted} nodes (Pages + Chunks).")

    def hard_reset(self):
        """Wipes both data and schema for a completely fresh start."""
        self.clear_data()
        self.clear_schema()
        print("Hard Reset Complete: The database is now empty and has no rules.")

    def get_all_publishers(self):
        """
        Returns a sorted list of all unique Publishers in the Knowledge Graph.
        """
        with self.driver.session() as session:
            # Match all Publisher nodes and return their 'name' property
            query = """
            MATCH (p:Publisher) 
            RETURN p.name AS name 
            ORDER BY name ASC
            """
            result = session.run(query)
            
            # Extract names from the result records
            publishers = [record["name"] for record in result]
            
            print(f"Found {len(publishers)} unique publishers.")
            return publishers

    def build_graph_relationships(self):
        """
        Runs post-processing to link Documents to shared Entities (Authors, Publishers, Years).
        This creates the 'Knowledge Graph' structure where paths form between papers.
        """
        with self.driver.session() as session:
            print("Starting Graph Linking Phase...")

            # 1. Unify Authors (Merge "Smith" from Paper A and "Smith" from Paper B)
            # Result: (Author)-[:WROTE]->(Document)
            print("   ...Unifying Author Nodes")
            session.run("""
            MATCH (d:Document) WHERE d.author_list IS NOT NULL
            UNWIND d.author_list AS name
            WITH d, trim(name) AS cleaned_name
            WHERE size(cleaned_name) > 1  // Skip empty/garbage names
            MERGE (a:Author {name: cleaned_name})
            MERGE (a)-[:WROTE]->(d)
            """)

            # 2. Unify Publishers (Merge "CSIRO" from multiple reports)
            # Result: (Publisher)-[:PUBLISHED]->(Document)
            print("   ...Unifying Publisher Nodes")
            session.run("""
            MATCH (d:Document) WHERE d.publisher_list IS NOT NULL
            UNWIND d.publisher_list AS pub_name
            WITH d, trim(pub_name) AS cleaned_pub
            WHERE size(cleaned_pub) > 1
            MERGE (p:Publisher {name: cleaned_pub})
            MERGE (p)-[:PUBLISHED]->(d)
            """)

            # 3. Time Hierarchy (Connect Documents to common Year Nodes)
            # Result: (Document)-[:PUBLISHED_IN]->(Year)
            # This allows queries like: "Match all papers connected to Year 2024"
            print("   ...Building Time Hierarchy")
            session.run("""
            MATCH (d:Document) 
            WHERE d.year IS NOT NULL AND d.year <> 0
            
            // FIX: Force year to be an Integer to ensure "2024" connects to 2024
            WITH d, toInteger(d.year) as year_val
            
            MERGE (y:Year {val: year_val})
            MERGE (d)-[:PUBLISHED_IN]->(y)
            """)

        print("Graph connections complete.")
   