# Overlap Methodology

Feature-only 64-bit pandas row hashes were built by streaming the complete train CSV. Test rows were streamed and classified as overlapping when their feature hash occurred in the train hash set. Labels were excluded. Hash collisions are theoretically possible.
