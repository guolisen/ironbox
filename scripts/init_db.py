#!/usr/bin/env python3
"""
Script to initialize the IronBox database using SQL scripts.
This is an alternative to the SQLAlchemy ORM approach used in the main application.
"""
import os
import sqlite3
import argparse
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "ironbox.db"
SQL_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "init_db.sql"


def init_db(db_path=DB_PATH, sql_script_path=SQL_SCRIPT_PATH):
    """
    Initialize the database using SQL scripts.
    
    Args:
        db_path: Path to the database file
        sql_script_path: Path to the SQL script file
    """
    # Ensure directories exist
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read SQL script
    with open(sql_script_path, "r") as f:
        sql_script = f.read()
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Execute SQL script
        cursor.executescript(sql_script)
        conn.commit()
        print(f"Database initialized successfully at {db_path}")
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Initialize the IronBox database")
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DB_PATH),
        help="Path to the database file",
    )
    parser.add_argument(
        "--sql-script",
        type=str,
        default=str(SQL_SCRIPT_PATH),
        help="Path to the SQL script file",
    )
    args = parser.parse_args()
    
    init_db(args.db_path, args.sql_script)


if __name__ == "__main__":
    main()
