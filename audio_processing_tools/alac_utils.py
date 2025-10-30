#pylint: disable=C0103
#pylint: disable=C0114
#pylint: disable=C0115

import struct
import os
import sys

# Constants from your C code
kMinCAFFPacketTableHeaderSize = 24
kALACFormatAppleLossless = 0x616C6163
kALACFormatLinearPCM = 1819304813
kALACFormatFlagsNativeEndian = 2
kALACDefaultFramesPerPacket = 128
kALACMaxEscapeHeaderBytes = 12


# Constants from your code
kMinCAFFPacketTableHeaderSize = 24
kALACDefaultFramesPerPacket = 128
kALACMaxEscapeHeaderBytes = 16  # or 12 if you prefer (match your code)

class AudioFormatDescription:
    def __init__(self, mSampleRate=0.0, mFormatID=0, mFormatFlags=0,
                 mBytesPerPacket=0, mFramesPerPacket=0, mBytesPerFrame=0,
                 mChannelsPerFrame=0, mBitsPerChannel=0, mReserved=0):
        self.mSampleRate = mSampleRate
        self.mFormatID = mFormatID
        self.mFormatFlags = mFormatFlags
        self.mBytesPerPacket = mBytesPerPacket
        self.mFramesPerPacket = mFramesPerPacket
        self.mBytesPerFrame = mBytesPerFrame
        self.mChannelsPerFrame = mChannelsPerFrame
        self.mBitsPerChannel = mBitsPerChannel
        self.mReserved = mReserved

class CAFPacketTableHeader:
    """
    Pythonic mirror of port_CAFPacketTableHeader in your C code.
    Typically has:
      uint64_t mNumberPackets
      uint64_t mNumberValidFrames
      uint32_t mPrimingFrames
      uint32_t mRemainderFrames
    """
    def __init__(self,
                 mNumberPackets=0,
                 mNumberValidFrames=0,
                 mPrimingFrames=0,
                 mRemainderFrames=0):
        self.mNumberPackets = mNumberPackets
        self.mNumberValidFrames = mNumberValidFrames
        self.mPrimingFrames = mPrimingFrames
        self.mRemainderFrames = mRemainderFrames

def build_base_packet_table(input_format, input_data_size):
    """
    Python equivalent of BuildBasePacketTable(AudioFormatDescription, \
          int32_t, int32_t*, port_CAFPacketTableHeader*)
    
    Fills out a CAFPacketTableHeader and returns:
      (packet_table_header, max_packet_table_size)
    """
    header = CAFPacketTableHeader()

    bytes_per_sample = (input_format.mBitsPerChannel >> 3) * input_format.mChannelsPerFrame

    header.mNumberValidFrames = input_data_size // bytes_per_sample
    header.mNumberPackets = header.mNumberValidFrames // kALACDefaultFramesPerPacket

    header.mPrimingFrames = 0

    remainder_frames = header.mNumberValidFrames - header.mNumberPackets \
        * kALACDefaultFramesPerPacket
    remainder_frames = kALACDefaultFramesPerPacket - remainder_frames
    if remainder_frames:
        header.mNumberPackets += 1
    header.mRemainderFrames = remainder_frames

    # Worst-case scenario for packet size
    the_max_packet_size = ((input_format.mBitsPerChannel >> 3) *
                           input_format.mChannelsPerFrame *
                           kALACDefaultFramesPerPacket) + kALACMaxEscapeHeaderBytes

    # Determine # of bytes used by each packet entry
    if the_max_packet_size < 16384:
        byte_size_table_entry = 2
    else:
        byte_size_table_entry = 3

    the_max_packet_table_size = byte_size_table_entry * header.mNumberPackets

    return header, the_max_packet_table_size


def write_caff_pakt_chunk_header(output_file, packet_table_header, packet_table_size):
    """
    Python equivalent of WriteCAFFpaktChunkHeader(FILE*, port_CAFPacketTableHeader*, uint32_t)
    
    The original code does a little-endian swap, writes 'pakt', chunk size, then the header.
    The minimal CAFF packet table header is 24 bytes (kMinCAFFPacketTableHeaderSize).
    """
    # We'll do the same little-endian swaps that your code does:
    #   mNumberPackets => 64-bit
    #   mNumberValidFrames => 64-bit
    #   mPrimingFrames => 32-bit
    #   mRemainderFrames => 32-bit
    np_be = struct.pack(">Q", packet_table_header.mNumberPackets)
    nvf_be = struct.pack(">Q", packet_table_header.mNumberValidFrames)
    pf_be  = struct.pack(">I", packet_table_header.mPrimingFrames)
    rf_be  = struct.pack(">I", packet_table_header.mRemainderFrames)

    # 'pakt' + 4 zero bytes + 4 bytes for packet_table_size
    # (Your code might do 'pakt' then 8 bytes, then the size in the next 4.
    #  We'll replicate exactly what we see:
    #    theBuffer[0..3] = 'pakt'
    #    theBuffer[4..7] = 0,0,0,0
    #    theBuffer[8..11] = packet_table_size
    # Then the 24-byte header
    header = bytearray(12)
    header[0:4] = b'pakt'
    # chunk size little-endian

    header[8] = (packet_table_size >> 24) & 0xFF
    header[9] = (packet_table_size >> 16) & 0xFF
    header[10] = (packet_table_size >> 8) & 0xFF
    header[11] = packet_table_size & 0xFF

    output_file.write(header)

    # Then write the 24-byte packet table header:
    # [0..7] mNumberPackets (little-endian 64)
    # [8..15] mNumberValidFrames (little-endian 64)
    # [16..19] mPrimingFrames (little-endian 32)
    # [20..23] mRemainderFrames (little-endian 32)
    output_file.write(np_be)
    output_file.write(nvf_be)
    output_file.write(pf_be)
    output_file.write(rf_be)


def write_caff_fcaff_chunk(output_file):
    """
    Matches the original C code:
      uint8_t theReadBuffer[8] = {'c','a','f','f', 0,1,0,0};
    Which in hex is: 63 61 66 66 00 01 00 00
    """
    output_file.write(b'\x63\x61\x66\x66\x00\x01\x00\x00')


def write_caff_desc_chunk(output_file, audio_format):
    """
    Replicates the original C code precisely:
    
      void WriteCAFFdescChunk(FILE * outputFile, AudioFormatDescription theOutputFormat)
      {
          // Setup the 12-byte header with 'desc'
          // theReadBuffer[11] = sizeof(port_CAFAudioDescription);
          // Then write the little-endian-swapped fields of the CAFAudioDescription struct.
      }
    """

    # 1) 12-byte header (theReadBuffer), with 'desc' and then 8 zeros, \
    # then the last byte = struct size.
    header = bytearray([
        ord('d'), ord('e'), ord('s'), ord('c'),
        0, 0, 0, 0,
        0, 0, 0, 0
    ])
    # In your original code, the last byte gets set to sizeof(port_CAFAudioDescription),
    # which you noted appears to be 32. So:
    header[11] = 32

    # Write these 12 bytes exactly.
    output_file.write(header)

    # 2) Write the port_CAFAudioDescription struct in little-endian. It is 32 bytes, laid out as:
    #
    #   Offset  Size  Field
    #   ------  ----  --------------------------------
    #   0       8     mSampleRate (double, little-endian)
    #   8       4     mFormatID   (uint32_t, little-endian)
    #   12      4     mFormatFlags (uint32_t, little-endian)
    #   16      4     mBytesPerPacket (uint32_t, little-endian)
    #   20      4     mFramesPerPacket (uint32_t, little-endian)
    #   24      4     mChannelsPerFrame (uint32_t, little-endian)
    #   28      4     mBitsPerChannel (uint32_t, little-endian)
    #
    # Total = 32 bytes
    desc = bytearray(32)

    # Sample rate as little-endian double
    struct.pack_into(">d", desc, 0, audio_format.mSampleRate)

    # Format ID as little-endian uint32
    struct.pack_into(">I", desc, 8,  audio_format.mFormatID)

    # Format flags
    struct.pack_into(">I", desc, 12, audio_format.mFormatFlags)

    # Bytes per packet
    struct.pack_into(">I", desc, 16, audio_format.mBytesPerPacket)

    # Frames per packet
    struct.pack_into(">I", desc, 20, audio_format.mFramesPerPacket)

    # Channels per frame
    struct.pack_into(">I", desc, 24, audio_format.mChannelsPerFrame)

    # Bits per channel
    struct.pack_into(">I", desc, 28, audio_format.mBitsPerChannel)
    print(audio_format.mBitsPerChannel)

    # Now write these 32 bytes
    output_file.write(desc)


def write_caff_data_chunk(output_file):
    """
    Matches the original C code:
      uint8_t theReadBuffer[16] = {
          'd','a','t','a', 0,0,0,0, 0,0,0,0, 0,0,0,1
      };
      fwrite(theReadBuffer,1,16,outputFile);
    Hex: 64 61 74 61  00 00 00 00  00 00 00 00  00 00 00 01
    """
    output_file.write(b'\x64\x61\x74\x61\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01')


def write_caff_kuki_chunk(output_file, in_cookie):
    """
    Matches the original C code:
      uint8_t thekukiHeaderBuffer[12] = {'k','u','k','i', 0,0,0,0, 0,0,0,0};
      thekukiHeaderBuffer[11] = inCookieSize;
      fwrite(thekukiHeaderBuffer,1,12,outputFile);
      fwrite(inCookie,1,inCookieSize,outputFile);
    """
    header = bytearray([ord('k'), ord('u'), ord('k'), ord('i'),
                        0, 0, 0, 0, 0, 0, 0, 0])
    cookie_size = len(in_cookie)
    # Place the size in the last byte
    header[11] = cookie_size & 0xFF
    output_file.write(header)
    output_file.write(in_cookie)


def read_ber_integer(in_buf, max_bytes):
    """
    Python equivalent of:
       uint32_t ReadBERInteger(uint8_t * theInputBuffer, int32_t * ioNumBytes)
    
    - `in_buf`: a 3- or 5-byte array read from file
    - `max_bytes`: the maximum number of bytes to consider (e.g., 2 or 3)
    
    Returns:
       (theAnswer, sizeUsed)
       where `theAnswer` is the decoded integer and `sizeUsed` is how many bytes
       were consumed from in_buf.
    """
    the_answer = 0
    size = 0
    while size < max_bytes and size < len(in_buf):
        the_data = in_buf[size]
        the_answer = (the_answer << 7) | (the_data & 0x7F)
        size += 1
        if size > 5:
            # In the original C code, if we exceed 5 bytes, we bail out.
            return 0, size
        # If the high bit is not set, we're done reading
        if (the_data & 0x80) == 0:
            break
    return the_answer, size


def find_caff_packet_table_start(input_file):
    """
    Matches original C code's logic to find the 'pakt' chunk.
    """
    current_position = input_file.tell()
    input_file.seek(8, os.SEEK_SET)
    pakt_pos, pakt_size = None, None

    while True:
        header = input_file.read(12)
        if len(header) < 12:
            break

        chunk_type = header[:4]  # raw bytes
        chunk_size = (header[8] << 24) | (header[9] << 16) | (header[10] << 8) | header[11]

        if chunk_type == b'pakt':
            # According to the C code:
            # *paktPos = ftell(inputFile) + kMinCAFFPacketTableHeaderSize;
            # *paktSize = chunkSize;
            pakt_pos = input_file.tell() + kMinCAFFPacketTableHeaderSize
            pakt_size = chunk_size
            break
        else:
            # Skip this chunk
            input_file.seek(chunk_size, os.SEEK_CUR)

    input_file.seek(current_position, os.SEEK_SET)
    return pakt_pos, pakt_size


def rearrange(input_file_name, output_file_name):
    """
    Python equivalent of:

      void rearrange(char * inputFileName, char * outputFileName)

    that *exactly* replicates the major chunk writes.
    """

    # Prepare input format
    input_format = AudioFormatDescription(
        mSampleRate=11162.0,
        mFormatID=kALACFormatLinearPCM,
        mFormatFlags=1,
        mBytesPerPacket=0,
        mFramesPerPacket=128,
        mBytesPerFrame=2,
        mChannelsPerFrame=1,
        mBitsPerChannel=16,
        mReserved=0
    )

    # Prepare an output format (as per original code logic, or you can adjust)
    output_format = AudioFormatDescription(
        mSampleRate=input_format.mSampleRate,
        mFormatID=kALACFormatAppleLossless,
        mFormatFlags=1,  # or kALACFormatFlagsNativeEndian, depending on your logic
        mFramesPerPacket=128,
        mChannelsPerFrame=input_format.mChannelsPerFrame,
        mBitsPerChannel=0,
        mBytesPerPacket=0,
        mBytesPerFrame=0
    )

    magic_cookie = b'\x00\x00\x00\x80\x00\x10\x28\x0a\x0e\x01\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x2b\x9a'

    try:
        with open(input_file_name, "rb") as f_in, open(output_file_name, "wb") as f_out:
            # 1) Write the 'caff' chunk
            write_caff_fcaff_chunk(f_out)

            # 2) Write the 'desc' chunk
            write_caff_desc_chunk(f_out, output_format)

            # 3) Magic cookie (kuki chunk) - just an example 128 bytes of zero
            write_caff_kuki_chunk(f_out, magic_cookie)

            # 4) We won't do the full packet-table logic here unless needed,
            #    but we can replicate a 'find' or skip.  The original code wrote a pakt chunk next,
            #    or we can skip it, depending on your original rearrange code.

            # Example: Build packet table from an "inputDataSize" = 245760
            packet_table_header, packet_table_size = build_base_packet_table(input_format, 245760)

            # Allocate the packet table entries (thePacketTableSize)
            packet_table_entries = bytearray(packet_table_size)  # zeroed

            # Then we add kMinCAFFPacketTableHeaderSize
            total_pakt_chunk_size = packet_table_size + kMinCAFFPacketTableHeaderSize

            write_caff_pakt_chunk_header(f_out, packet_table_header, total_pakt_chunk_size)

            # Record the position
            packetTablePos = f_out.tell()

            # Now subtract the header size so we only write the "entries"
            # This matches your line: thePacketTableSize -= kMinCAFFPacketTableHeaderSize;
            # But we already appended, so just write:
            f_out.write(packet_table_entries)


            # 5) Write 'data' chunk
            write_caff_data_chunk(f_out)

            # h) record dataPos
            dataPos = f_out.tell()

            # 6) Example data copying from input to output:
            #    The original code checked for 0xdecafbad, etc. Let's replicate that logic briefly:
            header_4 = f_in.read(4)
            if len(header_4) < 4:
                return

            # Compare [3], [2], [1], [0] with 0xDE, 0xCA, 0xFB, 0xAD
            if (header_4[3] == 0xDE and
                header_4[2] == 0xCA and
                header_4[1] == 0xFB and
                header_4[0] == 0xAD):
                # skip next 36 bytes => total 40 bytes from start
                f_in.seek(36, os.SEEK_CUR)
            else:
                # not valid => go back to start
                f_in.seek(0, os.SEEK_SET)

            # Now, in a loop, read 3 bytes => interpret as BER => read that many => write them
            while True:
                small_buf = f_in.read(3)
                if len(small_buf) < 3:
                    break

                the_read_size = 2
                theReadBytes, _ = read_ber_integer(small_buf, the_read_size)

                # Write small_buf[:small_buf[2]] at packetTablePos
                length_for_table = small_buf[2]
                f_out.seek(packetTablePos, os.SEEK_SET)
                f_out.write(small_buf[:length_for_table])
                packetTablePos += length_for_table

                # Read the payload
                data_buf = f_in.read(theReadBytes)
                if len(data_buf) < theReadBytes:
                    break

                # Write it at dataPos
                f_out.seek(dataPos, os.SEEK_SET)
                f_out.write(data_buf)
                dataPos += theReadBytes
    except IOError as e:
        print(f"Error opening or writing files: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_file.py input.caf output.caf")
        sys.exit(1)
    try:
        rearrange(sys.argv[1], sys.argv[2])
    except ValueError as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        sys.exit(1)
