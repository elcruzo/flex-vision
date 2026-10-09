"""Make a shareable SQLite copy without environment credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def sanitize(source,target):
    if target.exists():raise ValueError('Sanitized trace output must be new')
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as original,sqlite3.connect(target) as copy:
        original.backup(copy)
        copy.execute('PRAGMA secure_delete=ON')
        tables={row[0] for row in copy.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        environment_values=[]
        if 'TARGET_INFO_SYSTEM_ENV' in tables:
            environment_values=[row[0] for row in copy.execute('SELECT value FROM TARGET_INFO_SYSTEM_ENV') if row[0]]
            copy.execute('DELETE FROM TARGET_INFO_SYSTEM_ENV')
        redacted=0
        if 'StringIds' in tables:
            for index,value in copy.execute('SELECT id,value FROM StringIds').fetchall():
                credential=bool(re.search(r'rpa_[A-Za-z0-9_-]+',value))
                # Environment dumps can also contain other provider credentials.
                environment=any(secret in value for secret in environment_values if len(secret)>=16)
                if credential or environment:
                    copy.execute('UPDATE StringIds SET value=? WHERE id=?',('[redacted environment string]',index));redacted+=1
        copy.commit();copy.execute('VACUUM')
    return {'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'sanitized_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'redacted_strings':redacted,'policy':'remove environment rows and environment-bearing strings; preserve GPU events'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(sanitize(args.source,args.output)))


if __name__=='__main__':main()
