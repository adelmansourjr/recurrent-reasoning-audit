"""Replay the frozen follow-up with JSON-normalised mapping keys.

The frozen inference source returns integer option-label keys in one mapping.
JSON serialisation turns them into strings. This wrapper fixes comparison only;
it does not modify frozen inference, outcomes, gates or confidence intervals.
"""
import argparse
import json
from pathlib import Path
import selection_followup

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('results',nargs='?',type=Path,default=Path(__file__).with_name('followup_results.json'))
 args=parser.parse_args()
 actual=selection_followup.analyse()
 actual=json.loads(json.dumps(actual,sort_keys=True))
 expected=json.loads(args.results.read_text())
 assert actual==expected,'Frozen follow-up replay differs beyond JSON key normalisation'
 print('Verified fresh follow-up outcomes, pooling, costs, intervals and claim gates; JSON keys normalised.')
if __name__=='__main__':main()
