import struct
import wave


def read_chunk_header(f):
    chunk = f.read(8)
    if len(chunk) != 8:
        return None
    return struct.unpack("<4sI", chunk)


def read_chunk_data(f, size):
    data = f.read(size)
    if len(data) != size:
        raise Exception("Error reading chunk data")
    return data


def print_format(fmt):
    print("\n\nWAVE FORMAT INFO\n\n")
    print("Compression Code:       {}".format(fmt.get("comp_code")))
    print("Number of Channels:     {}".format(fmt.get("num_channels")))
    print("Sample Rate:            {}".format(fmt.get("sample_rate")))
    print("Average Bytes/Second:   {}".format(fmt.get("avg_bytes_per_sec")))
    print("Block Align:            {}".format(fmt.get("block_align")))
    print("Bits per Sample:        {}".format(fmt.get("bits_per_sample")))


def main(filename):
    with open(filename, "rb") as f:
        # Read RIFF header
        riff_header = read_chunk_header(f)
        if riff_header is None or riff_header[0] != b"RIFF":
            raise Exception("Error, file is NOT a RIFF file!")
        file_size = riff_header[1] + 8

        # Check WAVE format
        wave_format = f.read(4)
        if wave_format != b"WAVE":
            raise Exception("Error, file is not a WAVE file!")

        # Read chunks
        chunks = {}
        while True:
            chunk_header = read_chunk_header(f)
            if chunk_header is None:
                break
            chunk_id, chunk_size = chunk_header
            chunk_data = read_chunk_data(f, chunk_size)
            chunks[chunk_id] = chunk_data

        # Check for format and data chunks
        if b"fmt " not in chunks:
            raise Exception("Error, format chunk not found!")
        if b"data" not in chunks:
            raise Exception("Error, data chunk not found!")

        # Read format chunk
        fmt = struct.unpack("<HHIIHH", chunks[b"fmt "][:16])
        fmt = {
            "comp_code": fmt[0],
            "num_channels": fmt[1],
            "sample_rate": fmt[2],
            "avg_bytes_per_sec": fmt[3],
            "block_align": fmt[4],
            "bits_per_sample": fmt[5],
        }

        # Check for uncompressed PCM data
        if fmt["comp_code"] != 1:
            raise Exception("Error, this file does not contain uncompressed PCM data!")

        # Print format information
        print_format(fmt)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("\n\nUSAGE:  wavereader filename.wav\n\n")
    else:
        main(sys.argv[1])
