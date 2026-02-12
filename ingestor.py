import os
import uuid
import io
import scrapy
import hashlib
import logging
import requests
from multiprocessing import Process, Value
from urllib.parse import urlparse
from pathlib import Path
import json
import re

# Scrapy Imports
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.utils.project import get_project_settings

# PDF Imports
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

# Mute Logging
logging.getLogger('scrapy').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)



def store_in_neo4j(db_manager, file_info, chunks, vectors, doc_vector):
    """
    Writes Document, Chunks, AND Relationships (Authors/Years) immediately.
    """
    file_hash = file_info['hash']
    
    # We combine storage + linking into one atomic transaction
    query = """
    MERGE (d:Document {hash: $hash})
    SET d.title = $title,
        d.year = $year,
        d.full_date = $full_date,
        d.trust_level = $trust_level,
        d.source = $source,
        d.url = $url,   // Store the source URL explicitly
        d.author_list = $authors,
        d.publisher_list = $publishers,
        d.embedding = $doc_vec

    // 1. LINK AUTHORS (Immediate)
    FOREACH (name IN $authors | 
        MERGE (a:Author {name: trim(name)})
        MERGE (a)-[:WROTE]->(d)
    )

    // 2. LINK PUBLISHERS (Immediate)
    FOREACH (pub IN $publishers | 
        MERGE (p:Publisher {name: trim(pub)})
        MERGE (p)-[:PUBLISHED]->(d)
    )

    // 3. LINK YEAR (Immediate)
    MERGE (y:Year {val: toInteger($year)})
    MERGE (d)-[:PUBLISHED_IN]->(y)

    // 4. CHUNKS (Delete old, create new)
    WITH d
    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(old_c)
    DETACH DELETE old_c

    WITH d
    UNWIND $chunk_data AS chunk
    CREATE (c:Chunk {id: chunk.id})
    SET c.text = chunk.text, 
        c.embedding = chunk.vector, 
        c.page = chunk.page
    CREATE (d)-[:HAS_CHUNK]->(c)
    """

    chunk_data = [{
        "id": str(uuid.uuid4()), 
        "text": c.page_content, 
        "vector": v, 
        "page": c.metadata.get("page", 1)
    } for c, v in zip(chunks, vectors)]
    
    with db_manager.driver.session() as session:
        session.run(query, 
            hash=file_hash,
            title=file_info.get('title'),
            year=file_info.get('year'),
            full_date=file_info.get('full_date'),
            trust_level=file_info.get('trust_level'),
            source=file_info.get('source'),
            url=file_info.get('source_url'), # Pass URL here
            authors=file_info.get('authors'),
            publishers=file_info.get('publishers'),
            doc_vec=doc_vector, 
            chunk_data=chunk_data
        )
    print(f"Stored & Linked: '{file_info.get('title')}'")


class SmartIngestor:
    def __init__(self, db_manager, embed_url, embed_model, llm_url, llm_model, base_dir="./data"):
        self.db_manager = db_manager
        self.base_dir = Path(base_dir)

        self.EMBED_URL = embed_url
        self.EMBED_MODEL = embed_model

        self.LLM_URL = llm_url
        self.LLM_MODEL = llm_model

        self.TRUSTED_DOMAINS = ['.gov.au', '.csiro.au', '.edu.au', 'nature.com', 'ipcc.ch']
        self.TRUSTED_PUBLISHERS = [
            'Bureau of Meteorology', 'CSIRO'
        ]

    # ENTRY POINTS 

    def ingest_local_file(self, file_path):
        """Entry point for a single local file."""
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        file_hash = self.compute_hash_from_bytes(file_bytes)
        filename = os.path.basename(file_path)

        if self.db_manager.is_hash_present(file_hash):
            print(f"SKIP: '{filename}' (Already in DB)")
            return False

        file_info = {
            "path": str(file_path),
            "filename": filename,
            "hash": file_hash,
            "source": "local",      # Hardcoded for files
            "source_url": "local",  # Marker for trust check
            "source_type": "local"
        }
        
        return self._process_single_document(file_info, file_bytes=None)

    def ingest_from_stream(self, filename, file_bytes, source_url):
        file_hash = self.compute_hash_from_bytes(file_bytes)
        if self.db_manager.is_hash_present(file_hash):
            # Optional: Print that we are skipping
            # print(f"SKIP: {filename}")
            return False 

        try:
            domain = urlparse(source_url).netloc.replace("www.", "")
        except:
            domain = "unknown_web_source"

        # PRINT URL HERE
        print(f"INGESTING: {source_url}") 
        
        file_info = {
            "path": None,
            "filename": filename,
            "hash": file_hash,
            "source": domain,
            "source_url": source_url,
            "source_type": "web"
        }
        return self._process_single_document(file_info, file_bytes=file_bytes)

    # CORE PROCESSING 

    def _determine_trust_level(self, source_url, publisher_list):
        """
        Calculates Trust Score (1 = Highest, 2 = Standard).
        Logic: Match Domain OR Match Publisher Name.
        """
        # Check URL Domain
        if source_url and source_url != "local":
            if any(d in source_url.lower() for d in self.TRUSTED_DOMAINS):
                return 1 # High Trust
        
        # Check Publishers
        pub_str = " ".join(publisher_list).lower()
        if any(p.lower() in pub_str for p in self.TRUSTED_PUBLISHERS):
            return 1 # High Trust

        return 2 # Low Trust
    
    def _extract_metadata(self, raw_docs):
        """
        Uses a local LLM to extract semantic metadata from the document header.
        Scans the first 25 elements to handle long author lists and deep metadata.
        Returns: title, authors (humans), publishers (orgs/journals), year, full_date
        """
        if not raw_docs:
            return "Unknown Title", [], [], 0, None

        # 1. Prepare Context 
        # We join the text but keep it under ~5000 characters to fit in the context window
        combined_text = "\n".join([c.page_content for c in raw_docs[:25]])
        context_text = combined_text[:5000] 

        # 2. Strict JSON Prompt with UPDATED RULES
        system_prompt = """
        You are a Strict Metadata Normalization API. Your job is to extract and STANDARDIZE key details from document headers.
        
        OUTPUT RULES:
        1. Return VALID JSON only. Do not use markdown blocks (no ```json).
        2. JSON Structure:
        {
            "title": "String",
            "authors": ["String"], 
            "publishers": ["String"],
            "year": Integer,
            "full_date": "String"
        }
        
        FIELD DEFINITIONS:
        - "authors": List of HUMAN names who wrote THIS specific paper. 
            * Exclude organizations (move to publishers).
            * Exclude authors mentioned in citations (e.g., ignore "Smith (2020) argues...").
            - Format: "First Initial. Last Name" (e.g., "T. Lane", "A. King").
            - Remove titles like "Dr.", "Prof.", "PhD".
            - If a full list is not available, take the first 3 et al.
            * If no humans are listed, return [].
        - "publishers": List of all Government Agencies, or Journal Names who wrote/published THIS specific paper.
            * Exclude author's home institution (e.g. "\na School of Geography, Earth and Atmospheric Sciences, The University of Melbourne, Australia\n)
            * Exclude "Australian Government" and focus on getting the department name(s), i.e., just "Bureau of Meteorology".
            * Example: ["Bureau of Meteorology", "ELSEVIER", "CSIRO"].
            - **Canonicalization is MANDATORY.** Convert variations to their most common form:
                * "Commonwealth Scientific and Industrial Research Organisation" -> "CSIRO"
                * "CSIRO Publishing" -> "CSIRO"
                * "Australian Government Bureau of Meteorology" -> "Bureau of Meteorology"
                * "The Bureau of Meteorology" -> "Bureau of Meteorology"
                * "Dept of Climate Change..." -> "Department of Climate Change"
                * "Department of the Environment..." -> "Department of Environment"
            - Remove "Australian Government" prefix.
            * Return [] if none found
        - "year": The primary year associated with the document.
            * RULE 1 (Reporting Period): If the title implies a specific reporting cycle (e.g., "Annual Report 2022", "State of the Climate 2022"), use that year, even if published in 2023.
            * RULE 2 (Publication Date): For research papers or historical analyses (e.g., "Climate in 1900"), ALWAYS use the actual publication year (e.g., 2025), not the year discussed in the title.
            * Return 0 if not found.
        - "full_date": "DD Month YYYY" (prioritize 'Published' or 'Available online' dates). Return 0 if not found.

        3. NOISE CONTROL:
        - Do not include "The", "Inc", "Ltd" at the start/end of publisher names.
        - Do not include addresses or locations (e.g. "Melbourne, Australia").
        
        4. IMPORTANT: Output minified JSON (single line).
        """

        user_message = f"EXTRACT METADATA FROM THIS TEXT:\n\n{context_text}"

        # 3. Call LLM
        payload = {
            "model": self.LLM_MODEL, 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1, 
            "stream": False,
            "max_tokens": 1500 
        }

        try:
            response = requests.post(f"{self.LLM_URL}", json=payload)
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content'].strip()
            
            # 4. Clean & Parse JSON
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx]

            data = json.loads(content)
            
            # Extract fields
            title = data.get("title", "Unknown Title")
            authors = data.get("authors", [])
            publishers = data.get("publishers", [])  # New Field
            year = data.get("year", 0)
            full_date = data.get("full_date", None)
            
            # Validation: Year fallback
            if year == 0 and full_date:
                year_match = re.search(r"\b(19|20)\d{2}\b", full_date)
                if year_match:
                    year = int(year_match.group(1))

            return title, authors, publishers, year, full_date

        except json.JSONDecodeError as je:
            print(f"JSON Parsing Failed. Raw output:\n{content}")
            return "Metadata Parse Error", [], [], 0, None
            
        except Exception as e:
            print(f"Metadata Extraction Failed: {e}")
            # Fallback
            fallback_title = "Unknown Title"
            if raw_docs and hasattr(raw_docs[0], 'metadata') and raw_docs[0].metadata.get('category') == 'Title':
                fallback_title = raw_docs[0].page_content
            return fallback_title, [], [], 0, None


    def _process_single_document(self, file_info, file_bytes=None):
        print(f"   🔍 Analyzing Layout: {file_info['filename']}...")
        
        try:
            # LOAD
            if file_info['source_type'] == 'local':
                loader = UnstructuredLoader(
                    file_path=file_info['path'],
                    strategy="hi_res",
                    mode="elements",
                    chunking_strategy="by_title",
                    max_characters=1000,
                    combine_text_under_n_chars=200,
                )
                raw_elements = loader.load()
            else:
                with io.BytesIO(file_bytes) as pdf_stream:
                    loader = UnstructuredLoader(
                        file=pdf_stream,
                        metadata_filename=file_info['filename'],
                        strategy="hi_res",
                        mode="elements",
                        chunking_strategy="by_title",
                        max_characters=1000,
                        combine_text_under_n_chars=200,
                    )
                    raw_elements = loader.load()

        except Exception as e:
            print(f"OCR/Loading Failed: {e}")
            return False

        # EXTRACT METADATA
        meta_tuple = self._extract_metadata(raw_elements) 
        print(f"Found Metadata: {meta_tuple}")
        
        title, authors, publishers, year, full_date = meta_tuple
        
        # CALCULATE TRUST
        trust_level = self._determine_trust_level(file_info['source_url'], publishers)
        
        file_info.update({
            "title": title,
            "authors": authors,
            "publishers": publishers,
            "year": year if year else 0000,
            "full_date": full_date,
            "trust_level": trust_level
        })

        # EMBED IDENTITY
        identity_text = f"Title: {title}. "
        if authors: identity_text += f"Authored by: {', '.join(authors)}. "
        if publishers: identity_text += f"Published by: {', '.join(publishers)}. "
        if year: identity_text += f"Year: {year}."

        doc_vector = self._generate_embedding(identity_text)
        if not doc_vector: return False

        # PROCESS CHUNKS
        processed_chunks = []
        for element in raw_elements:
            content = element.page_content.strip()
            if not content: continue
            category = element.metadata.get("category", "Unknown")
            if category in ["Header", "Footer"]: continue

            processed_chunks.append(Document(
                page_content=content,
                metadata={
                    "page": element.metadata.get("page_number", 1),
                    "type": category,
                    "title": file_info['title'],
                    "year": file_info['year'],
                    "url": file_info.get('source_url', '')
                }
            ))

        # EMBED CHUNKS & STORE
        batch_texts = [c.page_content for c in processed_chunks]
        if batch_texts:
            print(f"Embedding {len(processed_chunks)} chunks (Trust Level: {trust_level})...")
            chunk_vectors = self._generate_embedding_batch(batch_texts)
            
            if chunk_vectors:
                # Pass updated file_info to storage
                store_in_neo4j(self.db_manager, file_info, processed_chunks, chunk_vectors, doc_vector)
                return True
        else:
            print(f"No text content found.")
            return False

    # UTILS (Unchanged mainly, simplified process_local_queue)
    
    def compute_hash_from_bytes(self, file_bytes):
        return hashlib.sha256(file_bytes).hexdigest()

    def get_file_hash(self, filepath):
        with open(filepath, "rb") as f:
            return self.compute_hash_from_bytes(f.read())
            
    def _generate_embedding(self, text):
        try:
            resp = requests.post(self.EMBED_URL, json={"model": self.EMBED_MODEL, "input": text})
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"Embedding Error: {e}")
            return None

    def _generate_embedding_batch(self, texts):
        try:
            resp = requests.post(self.EMBED_URL, json={"model": self.EMBED_MODEL, "input": texts})
            resp.raise_for_status()
            return [item["embedding"] for item in resp.json()["data"]]
        except Exception as e:
            print(f"Batch Embedding Error: {e}")
            return None

    def process_local_queue(self):
        """Scans local folder and ingests."""
        print(f"--- Scanning Local Directory: {self.base_dir} ---")
        to_process = []
        existing_hashes = self.db_manager.get_processed_hashes()
        
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    full_path = os.path.join(root, file)
                    file_hash = self.get_file_hash(full_path)
                    
                    if file_hash in existing_hashes: continue
                    to_process.append(full_path)
        
        print(f"Found {len(to_process)} new files.")
        for path in to_process:
            self.ingest_local_file(path)
            
        if to_process:
            print("\nRunning Graph Linker...")
            self.db_manager.build_graph_relationships()


class EduSpider(CrawlSpider):
    name = "edu_spider"
    
    def __init__(self, start_url=None, allowed_domain=None, *args, **kwargs):
        super(EduSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        
        # If input is 'www.bom.gov.au', allow 'bom.gov.au' (which covers reg.bom.gov.au, etc.)
        if allowed_domain.startswith("www."):
            self.root_domain = allowed_domain[4:]
        else:
            self.root_domain = allowed_domain
            
        # We allow the root domain. Scrapy automatically allows subdomains of this.
        self.allowed_domains = [self.root_domain]
        
        # STRICT BLOCKLIST
        # We explicitly block binary files that are useless for a text chatbot
        self.blocklist = [
            # Images
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg', 'ico', 'webp', 'tiff',
            # Archives & Executables
            'zip', 'tar', 'gz', 'rar', '7z', 'exe', 'msi', 'dmg', 'iso', 'bin',
            # Audio/Video
            'mp3', 'mp4', 'avi', 'mov', 'mkv', 'wav',
            # Data
            'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'csv', 'json', 'xml',
            # Web Code
            'css', 'js', 'less', 'scss', 'woff', 'woff2', 'ttf', 'eot'
        ]

        self.rules = (
            Rule(
                LinkExtractor(
                    allow_domains=self.allowed_domains,
                    deny_extensions=self.blocklist, 
                    # Empty 'allow' means "Allow Everything Else" (HTML, PDF, PHP, ASPX)
                    allow=(),
                ),
                callback='parse_item',
                follow=True
            ),
        )
        self._compile_rules()

    def parse_item(self, response):
        content_type = response.headers.get(b'Content-Type', b'').decode('utf-8').lower()
        url = response.url
        
        # CASE 1: PDF (Found via Header OR Extension)
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            yield {
                'type': 'pdf', 
                'url': url, 
                'body_bytes': response.body, 
                'source_domain': self.allowed_domains[0]
            }
            return

        # CASE 2: WEB PAGE (HTML)
        if 'text/html' in content_type:
            # Basic text extraction
            text_content = " ".join(response.css('body ::text').getall())
            clean_text = " ".join(text_content.split())
            title = response.css('title::text').get() or "Untitled Page"
            
            yield {
                'type': 'page', 
                'url': url, 
                'title': title, 
                'text': clean_text, 
                'source_domain': self.allowed_domains[0]
            }


class Neo4jPipeline:
    def __init__(self, db_manager, pdf_ingestor, pdf_counter):
        self.db_manager = db_manager
        self.pdf_ingestor = pdf_ingestor
        self.pdf_counter = pdf_counter

    def process_item(self, item, spider):
        if item['type'] == 'pdf':
            # INCREMENT COUNTER SAFELY
            with self.pdf_counter.get_lock():
                self.pdf_counter.value += 1
                
            filename = item['url'].split('/')[-1] or "downloaded.pdf"
            self.pdf_ingestor.ingest_from_stream(
                filename=filename, file_bytes=item['body_bytes'], source_url=item['url']
            )
        elif item['type'] == 'page':
            self.store_web_page(item)
        return item

    def store_web_page(self, item):
        query = """
        MERGE (p:WebPage {url: $url})
        SET p.title = $title, p.text_content = $text, p.domain = $domain, 
            p.source = $domain, p.trust_level = 1, p.last_crawled = datetime()
        """
        try:
            with self.db_manager.driver.session() as session:
                session.run(query, url=item['url'], title=item['title'], text=item['text'], domain=item['source_domain'])
        except Exception as e:
            print(f"DB Error: {e}")


def run_spider_process(start_url, allowed_domain, db_config, embed_url, embed_model, llm_url, llm_model, Neo4jManagerClass, pdf_counter):
    proc_db_manager = Neo4jManagerClass(db_config['uri'], db_config['user'], db_config['password'])
    proc_ingestor = SmartIngestor(proc_db_manager, embed_url, embed_model, llm_url, llm_model)

    safe_domain = allowed_domain.replace(".", "_")
    job_dir = f"crawls/{safe_domain}"

    settings = get_project_settings()
    settings.setdict({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ROBOTSTXT_OBEY': False,
        'CONCURRENT_REQUESTS': 16,
        'DEPTH_LIMIT': 5, 
        'COOKIES_ENABLED': False,
        'LOG_LEVEL': 'INFO',
        'JOBDIR': job_dir,
    })

    process = CrawlerProcess(settings)
    crawler = process.create_crawler(EduSpider)
    # Pass counter to pipeline
    pipeline = Neo4jPipeline(proc_db_manager, proc_ingestor, pdf_counter) 
    crawler.signals.connect(pipeline.process_item, signal=scrapy.signals.item_scraped)
    process.crawl(crawler, start_url=start_url, allowed_domain=allowed_domain)
    process.start()

# MAIN WRAPPER
class SmartURLIngestor:
    def __init__(self, db_config, embed_url, embed_model, llm_url, llm_model, Neo4jManagerClass):
        self.db_config = db_config
        self.embed_url = embed_url
        self.embed_model = embed_model
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.Neo4jManagerClass = Neo4jManagerClass
        self.local_db = Neo4jManagerClass(db_config['uri'], db_config['user'], db_config['password'])

    def ingest_domain(self, start_url):
        domain = urlparse(start_url).netloc.replace("www.", "")
        print(f"\nStarting Scrapy for domain: {domain}")
        self.reset_domain_data(domain)

        # Create Shared Counter (Integer initialized to 0)
        pdf_counter = Value('i', 0)

        p = Process(target=run_spider_process, args=(
            start_url, domain, self.db_config, self.embed_url, self.embed_model, 
            self.llm_url, self.llm_model, self.Neo4jManagerClass, pdf_counter
        ))
        p.start()
        p.join()
        
        print(f"Scrapy finished processing {domain}.")
        print(f"Total PDF Files Encountered & Processed: {pdf_counter.value}")

    def reset_domain_data(self, domain):
        print(f"Wiping old data for {domain}...")
        with self.local_db.driver.session() as session:
            session.run("MATCH (n) WHERE n.domain = $d OR n.source = $d DETACH DELETE n", d=domain)