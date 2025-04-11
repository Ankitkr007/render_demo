import sqlite3
import json
from typing import Optional

class DBHandler:
    def __init__(self, db_path="recruitment.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        
    def _init_db(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            raw_description TEXT NOT NULL,
            summary TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id INTEGER PRIMARY KEY,
            job_id INTEGER NOT NULL,
            cv_text TEXT NOT NULL,
            parsed_data TEXT NOT NULL,
            embedding BLOB NOT NULL,
            score REAL NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(job_id)
        )""")
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            email_id INTEGER PRIMARY KEY,
            candidate_id INTEGER NOT NULL,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )""")
    
    def create_job(self, title: str, raw_description: str, summary: dict, embedding: bytes) -> int:
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO jobs (title, raw_description, summary, embedding)
            VALUES (?, ?, ?, ?)
        """, (title, raw_description, json.dumps(summary), embedding.numpy().tobytes()))
        job_id = cur.lastrowid
        self.conn.commit()
        return job_id
    
    def create_candidate(self, job_id: int, cv_text: str, cv_data: dict, embedding: bytes, score: float) -> int:
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO candidates (job_id, cv_text, parsed_data, embedding, score)
            VALUES (?, ?, ?, ?, ?)
        """, (job_id, cv_text, json.dumps(cv_data), embedding.numpy().tobytes(), score))
        candidate_id = cur.lastrowid
        self.conn.commit()
        return candidate_id
    
    def create_email(self, candidate_id: int, content: str) -> int:
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO emails (candidate_id, content)
            VALUES (?, ?)
        """, (candidate_id, content))
        self.conn.commit()
        return cur.lastrowid