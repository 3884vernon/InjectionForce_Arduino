#!/usr/bin/env python3
"""
FASTQ Sequence Analysis Script - Summary Results Only
Analyzes genetic sequences from FASTQ files against a reference sequence
"""

import re
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class FastqRecord:
    """Represents a single FASTQ record"""
    header: str
    sequence: str
    plus_line: str
    quality: str


@dataclass
class AlignmentResult:
    """Represents alignment results between query and reference"""
    query_seq: str
    ref_seq: str
    percent_match: float
    overlaps: List[Tuple[int, int, str]]  # (start, end, sequence)
    insertions: List[Tuple[int, str]]  # (position, inserted_sequence)
    deletions: List[Tuple[int, str]]  # (position, deleted_sequence)


class FastqAnalyzer:
    def __init__(self, reference_file: str = ""):
        self.reference_sequence = ""
        self.sequences = []
        if reference_file:
            self.load_reference_sequence(reference_file)

    # === LOAD REFERENCE SEQUENCE FROM FILE ===
    def load_reference_sequence(self, filepath: str) -> str:
        """
        Load reference sequence from a text file
        Handles FASTA format or plain text
        """
        try:
            with open(filepath, 'r') as file:
                content = file.read().strip()

            # Handle FASTA format (lines starting with >)
            if content.startswith('>'):
                lines = content.split('\n')
                sequence_lines = []
                for line in lines:
                    if not line.startswith('>') and line.strip():
                        sequence_lines.append(line.strip())
                sequence = ''.join(sequence_lines)
            else:
                # Handle plain text (remove any whitespace/newlines)
                sequence = ''.join(content.split())

            # Validate sequence
            valid_bases = set('ATCGN')
            sequence = sequence.upper()

            if not all(base in valid_bases for base in sequence):
                print("Warning: Reference sequence contains invalid DNA bases")

            self.reference_sequence = sequence
            print(f"Loaded reference sequence: {len(sequence)} bp")

            return sequence

        except FileNotFoundError:
            print(f"Error: Reference file {filepath} not found")
            return ""
        except Exception as e:
            print(f"Error loading reference sequence: {e}")
            return ""

    # === LOAD FASTQ FILE ===
    def load_fastq_file(self, filepath: str) -> List[FastqRecord]:
        """
        Load and parse FASTQ file
        FASTQ format:
        @header
        sequence
        +
        quality_scores
        """
        records = []
        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()

            # Process every 4 lines as one record
            for i in range(0, len(lines), 4):
                if i + 3 < len(lines):
                    header = lines[i].strip()
                    sequence = lines[i + 1].strip()
                    plus_line = lines[i + 2].strip()
                    quality = lines[i + 3].strip()

                    record = FastqRecord(header, sequence, plus_line, quality)
                    records.append(record)

            print(f"Loaded {len(records)} sequences from FASTQ file")
            return records

        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return []
        except Exception as e:
            print(f"Error loading FASTQ file: {e}")
            return []

    # === FIND GENETIC SEQUENCES AND LOAD INTO LIST ===
    def extract_sequences(self, fastq_records: List[FastqRecord]) -> List[str]:
        """
        Extract DNA sequences from FASTQ records and validate them
        """
        sequences = []
        valid_bases = set('ATCGN')  # N represents unknown base

        for record in fastq_records:
            sequence = record.sequence.upper()

            # Validate sequence contains only valid DNA bases
            if all(base in valid_bases for base in sequence):
                sequences.append(sequence)
            else:
                print(f"Warning: Invalid bases found in sequence: {record.header}")

        self.sequences = sequences
        print(f"Extracted {len(sequences)} valid DNA sequences")
        return sequences

    # === MATCH WITH REFERENCE SEQUENCE ===
    def align_with_reference(self, query_sequence: str) -> AlignmentResult:
        """
        Perform simple alignment between query and reference sequence
        Uses a sliding window approach to find best alignment
        """
        if not self.reference_sequence:
            raise ValueError("Reference sequence not set")

        query = query_sequence.upper()
        reference = self.reference_sequence

        best_score = 0
        best_offset = 0

        # Try different alignments by sliding the query sequence
        for offset in range(-len(query), len(reference)):
            score = self._calculate_alignment_score(query, reference, offset)
            if score > best_score:
                best_score = score
                best_offset = offset

        # Generate alignment result
        return self._generate_alignment_result(query, reference, best_offset)

    def _calculate_alignment_score(self, query: str, reference: str, offset: int) -> int:
        """Calculate alignment score for given offset"""
        score = 0
        query_start = max(0, -offset)
        ref_start = max(0, offset)

        for i in range(min(len(query) - query_start, len(reference) - ref_start)):
            if query[query_start + i] == reference[ref_start + i]:
                score += 1

        return score

    def _generate_alignment_result(self, query: str, reference: str, offset: int) -> AlignmentResult:
        """Generate detailed alignment result"""
        query_start = max(0, -offset)
        ref_start = max(0, offset)

        # Determine alignment region
        align_length = min(len(query) - query_start, len(reference) - ref_start)

        overlaps = []
        insertions = []
        deletions = []
        matches = 0

        # Analyze overlapping region
        if align_length > 0:
            query_aligned = query[query_start:query_start + align_length]
            ref_aligned = reference[ref_start:ref_start + align_length]

            # Find continuous matching regions
            match_start = None
            for i in range(align_length):
                if query_aligned[i] == ref_aligned[i]:
                    matches += 1
                    if match_start is None:
                        match_start = i
                else:
                    if match_start is not None:
                        overlaps.append((
                            ref_start + match_start,
                            ref_start + i - 1,
                            ref_aligned[match_start:i]
                        ))
                        match_start = None

            # Handle final match region
            if match_start is not None:
                overlaps.append((
                    ref_start + match_start,
                    ref_start + align_length - 1,
                    ref_aligned[match_start:align_length]
                ))

        # Find insertions (query bases not in reference)
        if offset < 0:  # Query extends before reference
            insertions.append((0, query[:abs(offset)]))
        if query_start + align_length < len(query):  # Query extends after alignment
            insertions.append((
                ref_start + align_length,
                query[query_start + align_length:]
            ))

        # Find deletions (reference bases not in query)
        if offset > 0:  # Reference extends before query
            deletions.append((0, reference[:offset]))
        if ref_start + align_length < len(reference):  # Reference extends after alignment
            deletions.append((
                ref_start + align_length,
                reference[ref_start + align_length:]
            ))

        # Calculate percent match
        total_bases = max(len(query), len(reference))
        percent_match = (matches / total_bases) * 100 if total_bases > 0 else 0

        return AlignmentResult(
            query_seq=query,
            ref_seq=reference,
            percent_match=percent_match,
            overlaps=overlaps,
            insertions=insertions,
            deletions=deletions
        )

    def analyze_sequence_summary(self, query_sequence: str, seq_number: int = None) -> Dict:
        """Analyze sequence and return summary results only"""
        alignment = self.align_with_reference(query_sequence)

        # Calculate total overlapping base pairs
        total_overlap_bp = sum(end - start + 1 for start, end, _ in alignment.overlaps)

        summary = {
            'sequence_number': seq_number,
            'sequence': query_sequence,
            'sequence_length': len(query_sequence),
            'percent_match': alignment.percent_match,
            'overlapping_bp': total_overlap_bp,
            'total_insertions': sum(len(seq) for _, seq in alignment.insertions),
            'total_deletions': sum(len(seq) for _, seq in alignment.deletions)
        }

        return summary

    def print_summary_header(self):
        """Print header for summary table"""
        print(f"\n{'=' * 80}")
        print("FASTQ SEQUENCE ANALYSIS SUMMARY")
        print(f"{'=' * 80}")
        print(f"{'Seq#':<5} {'Length':<8} {'Match%':<8} {'Overlap BP':<12} {'Sequence Preview':<30}")
        print(f"{'-' * 80}")

    def print_summary_row(self, summary: Dict):
        """Print a single summary row"""
        seq_num = summary['sequence_number'] if summary['sequence_number'] else "N/A"
        length = summary['sequence_length']
        match_pct = f"{summary['percent_match']:.1f}%"
        overlap_bp = summary['overlapping_bp']
        seq_preview = summary['sequence'][:30] + "..." if len(summary['sequence']) > 30 else summary['sequence']

        print(f"{seq_num:<5} {length:<8} {match_pct:<8} {overlap_bp:<12} {seq_preview:<30}")


def main():
    """Example usage of the FASTQ analyzer with summary output only"""

    print("FASTQ Sequence Analysis Tool - Summary Mode")
    print("=" * 50)

    # Initialize analyzer with reference file
    reference_file = "/Users/timothyvernon/Downloads/GCaMP6s_Sequence.txt"  # Change this to your reference file path
    analyzer = FastqAnalyzer(reference_file)

    if not analyzer.reference_sequence:
        print("No reference sequence loaded. Using example sequence.")
        # Fallback to example reference
        analyzer.reference_sequence = "ATCGATCGATCGATCGATCGATCG"

    # Load FASTQ file
    fastq_file = "/Users/timothyvernon/Downloads/BZ77QQ_fastq/BZ77QQ_1_sample_1_G6.fastq"  # Change this to your FASTQ file path
    fastq_records = analyzer.load_fastq_file(fastq_file)
    sequences = analyzer.extract_sequences(fastq_records)

    print(f"Reference sequence length: {len(analyzer.reference_sequence)} bp")

    # Print summary header
    analyzer.print_summary_header()

    # Analyze sequences from FASTQ file
    if sequences:
        # Analyze all sequences (or set a limit if you want)
        num_to_analyze = (len(sequences))  # Analyze the total number of sequences in FASTQ file

        summaries = []
        for i in range(num_to_analyze):
            summary = analyzer.analyze_sequence_summary(sequences[i], i + 1)
            summaries.append(summary)
            analyzer.print_summary_row(summary)

        # Print overall statistics
        print(f"\n{'=' * 80}")
        print("OVERALL STATISTICS")
        print(f"{'=' * 80}")
        print(f"Total sequences analyzed: {len(summaries)}")
        print(f"Average match percentage: {sum(s['percent_match'] for s in summaries) / len(summaries):.1f}%")
        print(f"Average sequence length: {sum(s['sequence_length'] for s in summaries) / len(summaries):.0f} bp")
        print(f"Average overlapping base pairs: {sum(s['overlapping_bp'] for s in summaries) / len(summaries):.0f} bp")

        if len(sequences) > num_to_analyze:
            print(f"\nNote: Analyzed {num_to_analyze} out of {len(sequences)} total sequences.")
            print("To analyze more sequences, increase the 'num_to_analyze' variable.")
    else:
        print("No sequences found in FASTQ file. Using example sequences.")
        # Fallback to example sequences
        sample_sequences = [
            "ATCGATCGATCGATCGATCGATCG",  # Perfect match
            "ATCGATCGAAAAATCGATCGATCG",  # With insertion
            "ATCGATCGATCGATCG",  # With deletion
            "ATCGATCGTTCGATCGATCGATCG",  # With substitution
        ]

        for i, seq in enumerate(sample_sequences, 1):
            summary = analyzer.analyze_sequence_summary(seq, i)
            analyzer.print_summary_row(summary)


if __name__ == "__main__":
    main()