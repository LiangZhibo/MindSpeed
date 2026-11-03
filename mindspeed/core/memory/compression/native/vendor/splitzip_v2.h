
#pragma once
#include <acl/acl.h>
#include <array>
#include <cstddef>
#include <cstdint>
namespace splitzip_v2 {
constexpr uint32_t kFormat=0x535a4232, kVersion=2, kTile=8192, kBlocks=40, kEscapeCapacity=2048;
constexpr uint32_t kHeaderBytes=2048, kDenseThreshold=1170, kHotRatio=2;
constexpr uint64_t kMaxInputBytes=512ULL*1024*1024;
enum class Status { ok, invalid_argument, unsupported_size, insufficient_capacity, misaligned, overlap, invalid_codebook, invalid_packet, runtime_error };
const char* status_name(Status);
struct Buffer {void* data=nullptr;size_t bytes=0;};
struct Plan {
 uint32_t original_elements=0,padded_elements=0;
 size_t input_bytes=0,packet_capacity=0,sparse_bytes=655392,count_bytes=1536,table_bytes=544;
};
struct Codebook {std::array<uint16_t,272> lut{};uint32_t code_base=256;};
struct Workspace {Buffer sparse,counts,table;};
class PacketInfo {
 std::array<uint32_t,512> words_{};bool valid_=false;
 friend Status parse_header(const Plan&,const void*,size_t,size_t,PacketInfo*);
 public:
 bool valid()const{return valid_;}
 uint32_t stored_bytes()const{return words_[4];}
 uint32_t escapes()const{return words_[3];}
 uint32_t dense_tiles()const{return words_[181];}
 uint32_t sparse_escapes()const{return words_[187];}
 uint32_t max_core_sparse_escapes()const{uint32_t m=0;for(int i=336;i<376;i++)if(words_[i]>m)m=words_[i];return m;}
 uint32_t hotspot_cores()const;
 bool raw()const{return words_[176]!=0;}
 uint32_t max_core_escapes()const;
 uint32_t overflow_cores()const;
 const std::array<uint32_t,512>& header()const{return words_;}
};
Status make_plan(uint64_t elements,Plan*);
Status make_codebook(const uint16_t* sample,size_t elements,Codebook*);
Status validate_codebook(const Codebook&);
Status validate_encode(const Plan&,const Codebook&,Buffer input,Buffer packet,const Workspace&);
Status upload_codebook(const Codebook&,Buffer table,aclError* runtime_error=nullptr);
Status encode_async(const Plan&,const Codebook&,Buffer input,Buffer packet,const Workspace&,aclrtStream,aclError* runtime_error=nullptr);
Status parse_header(const Plan&,const void* host_header,size_t header_bytes,size_t packet_bytes,PacketInfo*);
Status read_packet_info(const Plan&,Buffer packet,Buffer pinned_header,aclrtStream,PacketInfo*,aclError* runtime_error=nullptr);
Status validate_decode(const Plan&,const PacketInfo&,Buffer packet,Buffer output);
Status decode_async(const Plan&,const PacketInfo&,Buffer packet,Buffer output,aclrtStream,aclError* runtime_error=nullptr);
}
