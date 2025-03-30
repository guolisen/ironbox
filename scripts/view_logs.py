#!/usr/bin/env python
"""
Script to view and filter IronBox API server logs.
"""
import argparse
import subprocess
import sys
import os
import re
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="View and filter IronBox API server logs")
    parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="DEBUG", help="Minimum log level to display")
    parser.add_argument("--module", help="Filter logs by module name (e.g., 'ironbox.api')")
    parser.add_argument("--since", help="Show logs since timestamp (format: YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--grep", help="Filter logs containing specific text")
    parser.add_argument("--tail", type=int, default=0, 
                        help="Show only the last N lines (0 for all)")
    parser.add_argument("--follow", "-f", action="store_true", 
                        help="Follow the log output (like tail -f)")
    return parser.parse_args()

def get_log_level_priority(level):
    """Get numeric priority for log level."""
    priorities = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }
    return priorities.get(level, 0)

def filter_log_line(line, args):
    """Filter a log line based on arguments."""
    # Check if line is a valid log line
    log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ([^ ]+) - ([A-Z]+) - (.+)'
    match = re.match(log_pattern, line)
    
    if not match:
        return False
    
    timestamp, module, level, message = match.groups()
    
    # Filter by log level
    if get_log_level_priority(level) < get_log_level_priority(args.level):
        return False
    
    # Filter by module
    if args.module and args.module not in module:
        return False
    
    # Filter by timestamp
    if args.since:
        try:
            log_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            since_time = datetime.strptime(args.since, '%Y-%m-%d %H:%M:%S')
            if log_time < since_time:
                return False
        except ValueError:
            # If timestamp parsing fails, don't filter
            pass
    
    # Filter by grep
    if args.grep and args.grep not in line:
        return False
    
    return True

def colorize_log_line(line):
    """Add color to log lines based on level."""
    log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ([^ ]+) - ([A-Z]+) - (.+)'
    match = re.match(log_pattern, line)
    
    if not match:
        return line
    
    timestamp, module, level, message = match.groups()
    
    # ANSI color codes
    colors = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m", # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m\033[37m", # White on red background
    }
    reset = "\033[0m"
    
    # Apply color to the level
    colored_level = f"{colors.get(level, '')}{level}{reset}"
    
    # Reconstruct the line with colored level
    return f"{timestamp} - {module} - {colored_level} - {message}"

def main():
    """Main function."""
    args = parse_args()
    
    # Check if running in a terminal that supports colors
    use_colors = sys.stdout.isatty()
    
    # Command to run the server and capture logs
    cmd = ["python", "-m", "ironbox.api.server"]
    
    if args.follow:
        # Run the server and stream logs
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        try:
            # Buffer for collecting lines for tail
            if args.tail > 0:
                buffer = []
            
            for line in iter(process.stdout.readline, ''):
                if filter_log_line(line.strip(), args):
                    if args.tail > 0:
                        buffer.append(line.strip())
                        if len(buffer) > args.tail:
                            buffer.pop(0)
                    else:
                        if use_colors:
                            print(colorize_log_line(line.strip()))
                        else:
                            print(line.strip())
            
            # Print buffered lines for tail
            if args.tail > 0:
                for line in buffer:
                    if use_colors:
                        print(colorize_log_line(line))
                    else:
                        print(line)
                        
        except KeyboardInterrupt:
            print("\nStopping log viewer...")
            process.terminate()
            process.wait()
    else:
        print("This script is designed to be used with --follow/-f flag to stream logs.")
        print("Starting the server with debug logging enabled...")
        print("Press Ctrl+C to stop.")
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nStopping server...")

if __name__ == "__main__":
    main()
