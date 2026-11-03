
#include "splitzip_v2.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <limits>
extern "C" void szb2_native_encode(uint32_t,uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
extern "C" void szb2_native_decode(uint32_t,void*,uint8_t*,uint8_t*);
namespace splitzip_v2 {
namespace {
uint64_t a32(uint64_t x){return (x+31)&~uint64_t(31);}
Status result(aclError e,aclError* out){if(out)*out=e;return e?Status::runtime_error:Status::ok;}
Status plan_ok(const Plan&p){Plan q;auto s=make_plan(p.original_elements,&q);if(s!=Status::ok)return s;return p.padded_elements==q.padded_elements&&p.input_bytes==q.input_bytes&&p.packet_capacity==q.packet_capacity&&p.sparse_bytes==q.sparse_bytes&&p.count_bytes==q.count_bytes&&p.table_bytes==q.table_bytes?Status::ok:Status::invalid_argument;}
Status buffer_ok(Buffer b,size_t required){if(!b.data)return Status::invalid_argument;if(b.bytes<required)return Status::insufficient_capacity;if(reinterpret_cast<uintptr_t>(b.data)%32)return Status::misaligned;if(b.bytes>UINTPTR_MAX-reinterpret_cast<uintptr_t>(b.data))return Status::invalid_argument;return Status::ok;}
bool intersect(Buffer a,Buffer b){auto x=reinterpret_cast<uintptr_t>(a.data),y=reinterpret_cast<uintptr_t>(b.data);return x<y+b.bytes&&y<x+a.bytes;}
}
const char*status_name(Status s){switch(s){case Status::ok:return "ok";case Status::invalid_argument:return "invalid_argument";case Status::unsupported_size:return "unsupported_size";case Status::insufficient_capacity:return "insufficient_capacity";case Status::misaligned:return "misaligned";case Status::overlap:return "overlap";case Status::invalid_codebook:return "invalid_codebook";case Status::invalid_packet:return "invalid_packet";default:return "runtime_error";}}
Status make_plan(uint64_t n,Plan*out){if(!out||!n)return Status::invalid_argument;if(n>kMaxInputBytes/2)return Status::unsupported_size;uint64_t padded=(n+kTile-1)/kTile*kTile;if(kHeaderBytes+padded*10>UINT32_MAX)return Status::unsupported_size;Plan p;p.original_elements=n;p.padded_elements=padded;p.input_bytes=padded*2;p.packet_capacity=kHeaderBytes+p.input_bytes;*out=p;return Status::ok;}
Status make_codebook(const uint16_t*s,size_t n,Codebook*out){if(!s||!n||!out)return Status::invalid_argument;std::array<uint64_t,256>h{};for(size_t i=0;i<n;i++)h[(s[i]>>7)&255]++;std::array<uint16_t,256>order;std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return h[a]>h[b];});std::sort(order.begin(),order.begin()+16);Codebook b;std::fill(b.lut.begin(),b.lut.begin()+256,16);for(int i=0;i<16;i++){b.lut[order[i]]=i;b.lut[256+i]=order[i];}if(b.lut[271]==b.lut[256]+15)b.code_base=b.lut[256];*out=b;return Status::ok;}
Status validate_codebook(const Codebook&b){std::array<uint16_t,256>expect;expect.fill(16);bool contig=true;for(int i=0;i<16;i++){auto e=b.lut[256+i];if(e>255||(i&&e<=b.lut[255+i]))return Status::invalid_codebook;expect[e]=i;if(e!=b.lut[256]+i)contig=false;}for(int i=0;i<256;i++)if(b.lut[i]!=expect[i])return Status::invalid_codebook;return b.code_base==(contig?b.lut[256]:256)?Status::ok:Status::invalid_codebook;}
Status validate_encode(const Plan&p,const Codebook&b,Buffer input,Buffer packet,const Workspace&w){auto s=plan_ok(p);if(s!=Status::ok)return s;s=validate_codebook(b);if(s!=Status::ok)return s;Buffer v[5]={input,packet,w.sparse,w.counts,w.table};size_t need[5]={p.input_bytes,p.packet_capacity,p.sparse_bytes,p.count_bytes,p.table_bytes};for(int i=0;i<5;i++){s=buffer_ok(v[i],need[i]);if(s!=Status::ok)return s;}for(int i=0;i<5;i++)for(int j=i+1;j<5;j++)if(intersect(v[i],v[j]))return Status::overlap;return Status::ok;}
Status upload_codebook(const Codebook&b,Buffer table,aclError*e){if(e)*e=ACL_SUCCESS;auto s=validate_codebook(b);if(s!=Status::ok)return s;s=buffer_ok(table,544);if(s!=Status::ok)return s;return result(aclrtMemcpy(table.data,table.bytes,b.lut.data(),544,ACL_MEMCPY_HOST_TO_DEVICE),e);}
Status encode_async(const Plan&p,const Codebook&b,Buffer input,Buffer packet,const Workspace&w,aclrtStream stream,aclError*e){if(e)*e=ACL_SUCCESS;auto s=validate_encode(p,b,input,packet,w);if(s!=Status::ok)return s;if(!stream)return Status::invalid_argument;size_t used=size_t(p.original_elements)*2;if(used<p.input_bytes){auto r=aclrtMemsetAsync(static_cast<uint8_t*>(input.data)+used,input.bytes-used,0,p.input_bytes-used,stream);if(r!=ACL_SUCCESS)return result(r,e);}szb2_native_encode(p.padded_elements,p.original_elements,stream,static_cast<uint8_t*>(input.data),static_cast<uint8_t*>(packet.data),static_cast<uint8_t*>(w.sparse.data),static_cast<uint8_t*>(w.counts.data),static_cast<uint8_t*>(w.table.data),b.code_base);return Status::ok;}

Status parse_header(const Plan&p,const void*data,size_t hb,size_t packet_bytes,PacketInfo*out){
 if(!out){return Status::invalid_argument;}
 *out=PacketInfo{};if(plan_ok(p)!=Status::ok||!data)return Status::invalid_argument;if(hb<kHeaderBytes)return Status::insufficient_capacity;
 std::array<uint32_t,512>h;memcpy(h.data(),data,kHeaderBytes);
 uint64_t tiles=p.padded_elements/kTile,per=(tiles+kBlocks-1)/kBlocks,modeStride=a32(per*4);
 if(h[0]!=kFormat||h[1]!=p.padded_elements||h[2]!=p.original_elements||h[177]!=kVersion||h[178]!=kTile||h[179]!=kBlocks||h[180]!=kEscapeCapacity||h[176]>1||h[184]!=tiles||h[185]!=kDenseThreshold||h[186]!=kHeaderBytes||h[188]!=kHotRatio||h[191]!=modeStride)return Status::invalid_packet;
 uint64_t all=0;for(int i=0;i<40;i++)all+=h[128+i];if(all!=h[3]||all>p.padded_elements)return Status::invalid_packet;
 uint64_t sparse=0,dense=0;
 for(uint64_t i=0;i<kBlocks;i++){
  uint64_t begin=i*per,nt=begin<tiles?std::min(per,tiles-begin):0,ne=nt*kTile,e=h[128+i],candidateEsc=h[416+i],candidates=h[456+i];
  if(e>ne||candidates>nt||candidateEsc>e||candidateEsc<candidates*(kDenseThreshold+1)||candidateEsc>candidates*kTile||(candidates==0&&candidateEsc!=0))return Status::invalid_packet;
  bool hot=e>kEscapeCapacity&&e*kBlocks>all*kHotRatio&&candidates>0;
  uint64_t d=hot?candidates:0,sp=e-(hot?candidateEsc:0);
  if(h[376+i]!=(hot?1u:0u)||h[336+i]!=sp||h[296+i]!=d||h[16+i]!=sparse||h[256+i]!=dense)return Status::invalid_packet;
  sparse+=sp;dense+=d;
 }
 uint64_t modeBase=kHeaderBytes+uint64_t(p.padded_elements)*3/2,cid=modeBase+(dense?modeStride*kBlocks:0),loc=cid+a32(sparse*4),val=loc+a32(sparse*2),denseBase=val+a32(sparse),packed=denseBase+dense*kTile,rawSize=kHeaderBytes+uint64_t(p.padded_elements)*2,used=std::min(packed,rawSize);
 if(h[5]!=cid||h[6]!=loc||h[7]!=val||h[4]!=used||h[176]!=(packed>rawSize)||h[181]!=dense||h[182]!=denseBase||h[183]!=modeBase||h[187]!=sparse||used>packet_bytes)return Status::invalid_packet;
 std::array<uint16_t,16>lut;memcpy(lut.data(),h.data()+8,32);for(int i=0;i<16;i++)if(lut[i]>255||(i&&lut[i]<=lut[i-1]))return Status::invalid_packet;
 for(int i=0;i<512;i++){bool occupied=i<56||(i>=128&&i<168)||(i>=176&&i<=188)||i==191||(i>=256&&i<496);if(!occupied&&h[i]!=0)return Status::invalid_packet;}
 out->words_=h;out->valid_=true;return Status::ok;
}
Status read_packet_info(const Plan&p,Buffer packet,Buffer host,aclrtStream stream,PacketInfo*out,aclError*e){if(e)*e=ACL_SUCCESS;if(!out)return Status::invalid_argument;*out=PacketInfo{};auto s=plan_ok(p);if(s!=Status::ok)return s;s=buffer_ok(packet,kHeaderBytes);if(s!=Status::ok)return s;s=buffer_ok(host,kHeaderBytes);if(s!=Status::ok)return s;if(!stream)return Status::invalid_argument;auto r=aclrtMemcpyAsync(host.data,host.bytes,packet.data,kHeaderBytes,ACL_MEMCPY_DEVICE_TO_HOST,stream);if(r)return result(r,e);r=aclrtSynchronizeStream(stream);if(r)return result(r,e);return parse_header(p,host.data,kHeaderBytes,packet.bytes,out);}
Status validate_decode(const Plan&p,const PacketInfo&info,Buffer packet,Buffer output){auto s=plan_ok(p);if(s!=Status::ok)return s;if(!info.valid())return Status::invalid_packet;PacketInfo q;s=parse_header(p,info.header().data(),kHeaderBytes,packet.bytes,&q);if(s!=Status::ok)return s;s=buffer_ok(packet,info.stored_bytes());if(s!=Status::ok)return s;s=buffer_ok(output,p.input_bytes);if(s!=Status::ok)return s;return intersect(packet,output)?Status::overlap:Status::ok;}
Status decode_async(const Plan&p,const PacketInfo&info,Buffer packet,Buffer output,aclrtStream stream,aclError*e){if(e)*e=ACL_SUCCESS;auto s=validate_decode(p,info,packet,output);if(s!=Status::ok)return s;if(!stream)return Status::invalid_argument;szb2_native_decode(p.padded_elements,stream,static_cast<uint8_t*>(packet.data),static_cast<uint8_t*>(output.data));return Status::ok;}
uint32_t PacketInfo::max_core_escapes()const{return *std::max_element(words_.begin()+128,words_.begin()+168);}
uint32_t PacketInfo::overflow_cores()const{return std::count_if(words_.begin()+336,words_.begin()+376,[](auto n){return n>kEscapeCapacity;});}
uint32_t PacketInfo::hotspot_cores()const{return std::count(words_.begin()+376,words_.begin()+416,1u);}

}
