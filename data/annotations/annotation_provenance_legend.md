# Annotation Provenance Legend

This benchmark is released as an annotation layer over public long-form speech recordings. The release does not include raw videos, audio files, converted WAV files, full ASR transcripts, or text-containing chunks.

## Fields

- `query_id`: Stable identifier for a heading-to-timestamp query.
- `source_id`: Stable identifier for the public recording in `data/manifest/source_manifest_release.csv`.
- `recording_alias`: Release-safe local filename alias. If users reconstruct local transcripts/chunks, they should use this alias consistently.
- `topic_heading`: Released benchmark heading/query.
- `timestamp`: Gold start timestamp in `HH:MM:SS` format.
- `timestamp_sec`: Gold timestamp converted to seconds.
- `split`: Evaluation split. The released experiment uses a test split.
- `heading_origin`: Coarse provenance of the topic heading.
- `timestamp_origin`: Coarse provenance of the timestamp.
- `verification_status`: Whether the label was checked by the authors.
- `release_status`: Confirms that this row is an annotation-only release.

## Provenance categories used in this release

- `mixed_public_seed_or_manual_normalized`: The heading may have been manually written, normalized from a public agenda/source description, or initialized from public chapter markers and then normalized.
- `mixed_public_seed_or_manual_verified`: The timestamp may have been manually identified or initialized from a public timeline/chapter/agenda and checked against the transcript/audio timeline where available.
- `manually_verified_or_author_verified`: The annotation was included after author verification. This field does not imply that all headings were written from scratch.

## Release policy

The authors' original annotation layer is released under the data license stated in the repository. This license does not grant any rights to third-party videos, audio, captions, transcripts, or source material. Users are responsible for obtaining lawful access to underlying source media before reconstructing transcripts or chunks.
