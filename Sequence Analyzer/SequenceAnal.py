#!/usr/bin/env python3
"""
jGCaMP7 FASTQ Sequence Search Tool
Searches for jGCaMP7 sequences in FASTQ files with comprehensive analysis

FOR PYCHARM USERS:
1. Edit the 'fastq_file' variable in the main() function to point to your FASTQ file
2. Optionally adjust 'min_overlap' and 'output_file' settings
3. Run the script directly in PyCharm
"""

import gzip
from collections import defaultdict
import re
from datetime import datetime


def reverse_complement(seq):
    """Return reverse complement of DNA sequence"""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, base) for base in reversed(seq.upper()))


def read_fastq(filename):
    """Generator to read FASTQ file (handles gzipped files)"""
    if filename.endswith('.gz'):
        file_handle = gzip.open(filename, 'rt')
    else:
        file_handle = open(filename, 'r')

    try:
        while True:
            header = file_handle.readline().strip()
            if not header:
                break
            sequence = file_handle.readline().strip()
            plus = file_handle.readline().strip()
            quality = file_handle.readline().strip()

            if header.startswith('@'):
                yield header, sequence, quality
    finally:
        file_handle.close()


def find_overlaps(sequence, target, min_overlap=20):
    """Find overlapping matches between sequence and target"""
    matches = []
    seq_upper = sequence.upper()
    target_upper = target.upper()

    # Find all possible overlaps
    for i in range(len(seq_upper) - min_overlap + 1):
        for j in range(len(target_upper) - min_overlap + 1):
            # Check if substring of sequence matches substring of target
            max_len = min(len(seq_upper) - i, len(target_upper) - j)
            for k in range(min_overlap, max_len + 1):
                if seq_upper[i:i + k] == target_upper[j:j + k]:
                    matches.append({
                        'seq_start': i,
                        'seq_end': i + k,
                        'target_start': j,
                        'target_end': j + k,
                        'length': k,
                        'match_seq': seq_upper[i:i + k]
                    })

    # Remove redundant matches (keep longest)
    if matches:
        matches.sort(key=lambda x: x['length'], reverse=True)
        filtered_matches = [matches[0]]
        for match in matches[1:]:
            # Check if this match overlaps with any existing match
            overlap_found = False
            for existing in filtered_matches:
                if (match['seq_start'] < existing['seq_end'] and
                        match['seq_end'] > existing['seq_start']):
                    overlap_found = True
                    break
            if not overlap_found:
                filtered_matches.append(match)
        return filtered_matches

    return matches


def analyze_fastq(fastq_file, target_sequence, min_overlap=20, output_file=None):
    """Analyze FASTQ file for jGCaMP7 sequences"""

    # jGCaMP7 sequence provided
    jgcamp7_seq = target_sequence.upper()
    jgcamp7_rc = reverse_complement(jgcamp7_seq)

    print(f"Starting analysis of {fastq_file}")
    print(f"Target sequence length: {len(jgcamp7_seq)} bp")
    print(f"Minimum overlap length: {min_overlap} bp")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Statistics
    total_reads = 0
    reads_with_matches = 0
    forward_matches = 0
    reverse_matches = 0
    all_matches = []
    coverage_map = defaultdict(int)  # Position -> count

    # Process FASTQ file
    for header, sequence, quality in read_fastq(fastq_file):
        total_reads += 1

        if total_reads % 10000 == 0:
            print(f"Processed {total_reads} reads...")

        # Search forward strand
        forward_overlaps = find_overlaps(sequence, jgcamp7_seq, min_overlap)

        # Search reverse strand
        reverse_overlaps = find_overlaps(sequence, jgcamp7_rc, min_overlap)

        # Record matches
        read_matches = []

        for match in forward_overlaps:
            forward_matches += 1
            match['strand'] = 'forward'
            match['read_id'] = header
            match['read_seq'] = sequence
            read_matches.append(match)

            # Update coverage map
            for pos in range(match['target_start'], match['target_end']):
                coverage_map[pos] += 1

        for match in reverse_overlaps:
            reverse_matches += 1
            match['strand'] = 'reverse'
            match['read_id'] = header
            match['read_seq'] = sequence
            read_matches.append(match)

            # Update coverage map for reverse complement
            for pos in range(match['target_start'], match['target_end']):
                coverage_map[len(jgcamp7_seq) - pos - 1] += 1

        if read_matches:
            reads_with_matches += 1
            all_matches.extend(read_matches)

    # Generate report
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total reads processed: {total_reads:,}")
    print(f"Reads with jGCaMP7 matches: {reads_with_matches:,}")
    print(f"Percentage of reads with matches: {(reads_with_matches / total_reads) * 100:.2f}%")
    print(f"Total forward strand matches: {forward_matches:,}")
    print(f"Total reverse strand matches: {reverse_matches:,}")
    print(f"Total matches found: {len(all_matches):,}")

    if all_matches:
        print("\n" + "=" * 60)
        print("DETAILED MATCH ANALYSIS")
        print("=" * 60)

        # Sort matches by length
        all_matches.sort(key=lambda x: x['length'], reverse=True)

        print(f"Longest match: {all_matches[0]['length']} bp")
        print(f"Shortest match: {all_matches[-1]['length']} bp")
        print(f"Average match length: {sum(m['length'] for m in all_matches) / len(all_matches):.1f} bp")

        # Coverage analysis
        covered_positions = len([pos for pos, count in coverage_map.items() if count > 0])
        coverage_percentage = (covered_positions / len(jgcamp7_seq)) * 100

        print(f"\nCoverage Analysis:")
        print(f"Positions covered: {covered_positions}/{len(jgcamp7_seq)} ({coverage_percentage:.1f}%)")
        print(f"Maximum coverage depth: {max(coverage_map.values()) if coverage_map else 0}x")
        print(f"Average coverage depth: {sum(coverage_map.values()) / len(jgcamp7_seq):.2f}x")

        # Show top matches
        print(f"\nTop 10 longest matches:")
        print("-" * 80)
        for i, match in enumerate(all_matches[:10]):
            print(f"{i + 1:2d}. Length: {match['length']:3d} bp | Strand: {match['strand']:7s} | "
                  f"Target pos: {match['target_start']:4d}-{match['target_end']:4d} | "
                  f"Read: {match['read_id'][:50]}...")

        # Coverage gaps
        gaps = []
        gap_start = None
        for pos in range(len(jgcamp7_seq)):
            if coverage_map[pos] == 0:
                if gap_start is None:
                    gap_start = pos
            else:
                if gap_start is not None:
                    gaps.append((gap_start, pos - 1))
                    gap_start = None

        if gap_start is not None:
            gaps.append((gap_start, len(jgcamp7_seq) - 1))

        if gaps:
            print(f"\nUncovered regions ({len(gaps)} gaps):")
            print("-" * 40)
            for start, end in gaps[:10]:  # Show first 10 gaps
                print(f"Position {start:4d}-{end:4d} ({end - start + 1:3d} bp)")
            if len(gaps) > 10:
                print(f"... and {len(gaps) - 10} more gaps")

    # Write detailed results to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"jGCaMP7 FASTQ Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {fastq_file}\n")
            f.write(f"Target sequence length: {len(jgcamp7_seq)} bp\n\n")

            f.write(f"SUMMARY:\n")
            f.write(f"Total reads: {total_reads:,}\n")
            f.write(f"Reads with matches: {reads_with_matches:,}\n")
            f.write(f"Forward matches: {forward_matches:,}\n")
            f.write(f"Reverse matches: {reverse_matches:,}\n\n")

            f.write(f"DETAILED MATCHES:\n")
            f.write(
                f"{'Read ID':<50} {'Strand':<8} {'Length':<6} {'Target Pos':<12} {'Read Pos':<10} {'Match Sequence':<50}\n")
            f.write("-" * 150 + "\n")

            for match in all_matches:
                f.write(f"{match['read_id'][:49]:<50} {match['strand']:<8} {match['length']:<6} "
                        f"{match['target_start']}-{match['target_end']:<8} {match['seq_start']}-{match['seq_end']:<8} "
                        f"{match['match_seq'][:49]:<50}\n")

        print(f"\nDetailed results written to: {output_file}")


def main():
    # Configuration - EDIT THESE PATHS FOR YOUR FILES
    fastq_file = "/Users/timothyvernon/Downloads/BZ77QQ_fastq/BZ77QQ_2_sample_2-G7_lib.fastq"  # <-- CHANGE THIS TO YOUR FASTQ FILE PATH
    min_overlap = 20  # Minimum overlap length
    output_file = "jgcamp7_results.txt"  # Output file (optional, set to None to skip)

    # jGCaMP7 sequence
    jgcamp7_sequence = "atgggttctcatcatcatcatcatcatggtatggctagcatgactggtggacagcaaatgggtcgggatctgtacgacgatgacgataaggatctcgccaccatggtcgactcatcacgtcgtaagtggaataagacaggtcacgcagtcagagtgataggtcggctgagctcactcgagaacgtctatatcaaggccgacaagcagaagaacggcatcaaggcgaacttccacatccgccacaacatcgaggacggcggcgtgcagctcgcctaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcgtgcagtccaaactttcgaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagggcggtaccggagggagcatggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggtgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacatccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagcttaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaaccttcctgaccaactgactgaagagcagatcgcagaatttaaagagcttttctccctatttgacaaggacggggatgggacaataacaaccaaggagctggggacggtgatgcggtctctggggcagaaccccacagaagcagagctgcaggacatgatcaatgaagtagatgccgacggtgacggcacaatcgacttccctgagttcctgacaatgatggcaagaaaaatgaaatacagggacacggaagaagaaattagagaagcgttcggtgtgtttgataaggatggcaatggctacatcagtgcagcagagcttcgccacgtgatgacaaaccttggagagaagttaacagatgaagaggttgatgaaatgatcagggaagcagacatcgatggggatggtcaggtaaactacgaagagtttgtacaaatgatgacagcgaag"

    # Check if file exists
    import os
    if not os.path.exists(fastq_file):
        print(f"ERROR: File not found: {fastq_file}")
        print(
            "Please update the 'fastq_file' variable in the main() function with the correct path to your FASTQ file.")
        return

    analyze_fastq(fastq_file, jgcamp7_sequence, min_overlap, output_file)


if __name__ == "__main__":
    main()