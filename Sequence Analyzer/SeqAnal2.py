#!/usr/bin/env python3

import re
import gzip
import os
import time
from collections import defaultdict
from datetime import datetime

def reverse_complement(seq):
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq.upper()))


def read_fastq(filename):
    """Generator to read sequences from a FASTQ file."""
    open_func = gzip.open if filename.endswith('.gz') else open
    with open_func(filename, 'rt') as fh:
        while True:
            header = fh.readline().strip()
            if not header:
                break
            seq = fh.readline().strip()
            fh.readline()  # skip '+'
            fh.readline()  # skip quality
            yield header, seq


def count_fastq_reads(filename):
    """Count number of reads in a FASTQ file (4 lines per read)."""
    open_func = gzip.open if filename.endswith('.gz') else open
    with open_func(filename, 'rt') as fh:
        return sum(1 for _ in fh) // 4


def find_kmer_matches(sequence, target, min_overlap):
    """
    Match k-mers of a minimum overlap length between read and target.
    Returns list of match dictionaries with position info.
    """
    seq = sequence.upper()
    target = target.upper()
    kmer_matches = []

    for i in range(len(seq) - min_overlap + 1):
        kmer = seq[i:i+min_overlap]
        for match in re.finditer(kmer, target):
            kmer_matches.append({
                'seq_start': i,
                'seq_end': i + min_overlap,
                'target_start': match.start(),
                'target_end': match.start() + min_overlap,
                'length': min_overlap,
                'match_seq': kmer
            })

    return kmer_matches


def analyze_fastq(fastq_file, target_sequence, min_overlap=20, output_file=None):
    """Main analysis function."""
    target_seq = target_sequence.upper()
    target_rc = reverse_complement(target_seq)

    total_reads = count_fastq_reads(fastq_file)
    print(f"\nAnalyzing: {fastq_file}")
    print(f"Target length: {len(target_seq)} bp | Min overlap: {min_overlap} bp")
    print(f"Total reads: {total_reads:,}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    start_time = time.time()
    total_processed = 0
    reads_with_hits = 0
    matches = []
    coverage = defaultdict(int)

    for header, seq in read_fastq(fastq_file):
        total_processed += 1
        forward_hits = find_kmer_matches(seq, target_seq, min_overlap)
        reverse_hits = find_kmer_matches(seq, target_rc, min_overlap)

        read_matches = []

        for hit in forward_hits:
            hit.update({'strand': 'forward', 'read_id': header})
            read_matches.append(hit)
            for i in range(hit['target_start'], hit['target_end']):
                coverage[i] += 1

        for hit in reverse_hits:
            hit.update({'strand': 'reverse', 'read_id': header})
            read_matches.append(hit)
            for i in range(hit['target_start'], hit['target_end']):
                coverage[i] += 1

        if read_matches:
            reads_with_hits += 1
            matches.extend(read_matches)

        if total_processed % 100 == 0:
            elapsed = time.time() - start_time
            rate = total_processed / elapsed
            print(f"\rProcessed {total_processed:,}/{total_reads:,} reads "
                  f"({(total_processed/total_reads)*100:.1f}%) | "
                  f"Rate: {rate:.1f} reads/s | Matches: {reads_with_hits:,}", end='')

    print(f"\n\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"Total reads processed:   {total_processed:,}")
    print(f"Reads with matches:      {reads_with_hits:,}")
    print(f"Total individual matches: {len(matches):,}")

    if matches:
        match_lengths = [m['length'] for m in matches]
        print(f"\nLongest match: {max(match_lengths)} bp")
        print(f"Shortest match: {min(match_lengths)} bp")
        print(f"Avg. match length: {sum(match_lengths)/len(match_lengths):.1f} bp")

        covered_positions = len([pos for pos in coverage if coverage[pos] > 0])
        coverage_pct = (covered_positions / len(target_seq)) * 100
        max_depth = max(coverage.values())
        avg_depth = sum(coverage.values()) / len(target_seq)

        print(f"\nTarget coverage: {covered_positions}/{len(target_seq)} positions "
              f"({coverage_pct:.1f}%)")
        print(f"Max coverage depth: {max_depth}x")
        print(f"Avg coverage depth: {avg_depth:.2f}x")

    if output_file:
        with open(output_file, 'w') as f:
            for match in matches:
                f.write(f"{match['read_id']}\t{match['strand']}\t{match['length']}\t"
                        f"{match['target_start']}-{match['target_end']}\t{match['match_seq']}\n")
        print(f"\nResults saved to: {output_file}")


def main():
    # 🔧 Edit these paths and sequences before running
    fastq_file = "/Users/timothyvernon/Downloads/BZ77QQ_fastq/BZ77QQ_2_sample_2-G7_lib.fastq"
    output_file = "jgcamp7_kmer_results.txt"
    min_overlap = 20

    jgcamp7_sequence = "atgggttctcatcatcatcatcat..."  # (paste full plasmid sequence here)

    if not os.path.exists(fastq_file):
        print(f"ERROR: FASTQ file not found: {fastq_file}")
        return

    analyze_fastq(fastq_file, jgcamp7_sequence, min_overlap, output_file)


if __name__ == "__main__":
    main()
