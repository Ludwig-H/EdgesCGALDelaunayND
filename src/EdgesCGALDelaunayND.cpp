// EdgesCGALDelaunayND.cpp (robust, ND + 3D fast-path)
// Read points from .npy (N,d) float32/float64, compute Delaunay edges, write .npy (M,2) int64
// ND path uses CGAL dD (Epick_d). If d==3 and TBB is available at build time, use 3D Delaunay with Parallel_tag.

#include <Eigen/Core>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_data_structure_3.h>

#include <CGAL/Epick_d.h>
#include <CGAL/Delaunay_triangulation.h>
#include <CGAL/Triangulation_data_structure.h>
#include <CGAL/Triangulation_vertex.h>
#include <CGAL/Triangulation_full_cell.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/Spatial_sort_traits_adapter_d.h>
#include <CGAL/property_map.h>
#include <CGAL/spatial_sort.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ---------------- NPY I/O (v1.0/v2.0, C-order) ----------------
struct NpyHeader {
  std::string descr;
  bool fortran_order=false;
  std::vector<size_t> shape;
  int major=1, minor=0;
  size_t header_len=0, data_offset=0;
};

static inline void die(const std::string& m){ std::cerr<<"[error] "<<m<<"\n"; std::exit(1); }

static NpyHeader parse_npy_header(std::istream& is){
  NpyHeader h;
  char magic[6]; if(!is.read(magic,6)) die("Not enough bytes for magic");
  if(!(magic[0]==char(0x93)&&magic[1]=='N'&&magic[2]=='U'&&magic[3]=='M'&&magic[4]=='P'&&magic[5]=='Y')) die("Bad .npy magic");
  unsigned char ver[2]; if(!is.read((char*)ver,2)) die("No version");
  h.major=ver[0]; h.minor=ver[1];
  uint16_t h16=0; uint32_t h32=0;
  if(h.major==1){ if(!is.read((char*)&h16,2)) die("No header len v1"); h.header_len=h16; }
  else if(h.major==2){ if(!is.read((char*)&h32,4)) die("No header len v2"); h.header_len=h32; }
  else die("Unsupported .npy version");
  std::string hdr(h.header_len,'\0'); if(!is.read(&hdr[0],h.header_len)) die("Header read failed");
  for(char& c:hdr) if(c=='"') c='\'';
  h.data_offset=(h.major==1?6+2+2:6+2+4)+h.header_len;

  auto find_between=[&](const std::string& key)->std::string{
    auto pos=hdr.find("'"+key+"'"); if(pos==std::string::npos) die("Missing key "+key);
    auto colon=hdr.find(':',pos); auto s=hdr.find('\'',colon); auto e=hdr.find('\'',s+1);
    if(colon==std::string::npos||s==std::string::npos||e==std::string::npos) die("Malformed header "+key);
    return hdr.substr(s+1,e-s-1);
  };
  h.descr=find_between("descr");
  {
    auto pos=hdr.find("'fortran_order'"); if(pos==std::string::npos) die("Missing fortran_order");
    auto colon=hdr.find(':',pos); auto comma=hdr.find(',',colon);
    auto val=hdr.substr(colon+1,comma-colon-1);
    auto l=val.find_first_not_of(" \t"); auto r=val.find_last_not_of(" \t"); if(l!=std::string::npos) val=val.substr(l,r-l+1);
    h.fortran_order = val.find("True")!=std::string::npos || val.find("true")!=std::string::npos;
  }
  {
    auto pos=hdr.find("'shape'"); if(pos==std::string::npos) die("Missing shape");
    auto lp=hdr.find('(',pos), rp=hdr.find(')',lp);
    if(lp==std::string::npos||rp==std::string::npos) die("Malformed shape tuple");
    auto inner=hdr.substr(lp+1,rp-lp-1); std::stringstream ss(inner);
    while(ss.good()){
      std::string tok; std::getline(ss,tok,',');
      auto a=tok.find_first_not_of(" \t"); auto b=tok.find_last_not_of(" \t");
      if(a==std::string::npos) continue; tok=tok.substr(a,b-a+1);
      char* endp=nullptr; long long v=strtoll(tok.c_str(),&endp,10);
      if(endp==tok.c_str()) continue; if(v<0) die("Negative dim in shape");
      h.shape.push_back((size_t)v);
    }
  }
  return h;
}

template<class T>
static void read_npy_matrix(const std::string& path, std::vector<T>& out, size_t& nrows, int& ncols){
  std::ifstream f(path, std::ios::binary); if(!f) die("Cannot open input "+path);
  NpyHeader h=parse_npy_header(f);
  if(h.fortran_order) die("Fortran-order not supported; save as C-order");
  if(h.shape.size()!=2) die("Expected 2D matrix (N,d)");
  nrows=h.shape[0]; ncols=(int)h.shape[1]; size_t count=nrows*(size_t)ncols;

  if(h.descr.find("f4")!=std::string::npos){
    std::vector<float> tmp(count); f.read((char*)tmp.data(), tmp.size()*sizeof(float)); if(!f) die("Read f4 payload failed");
    out.resize(count); for(size_t i=0;i<count;++i) out[i]=(T)tmp[i];
  }else if(h.descr.find("f8")!=std::string::npos){
    std::vector<double> tmp(count); f.read((char*)tmp.data(), tmp.size()*sizeof(double)); if(!f) die("Read f8 payload failed");
    out.resize(count); for(size_t i=0;i<count;++i) out[i]=(T)tmp[i];
  }else die("Unsupported dtype '"+h.descr+"', use float32/float64");
}

static void write_npy_edges_i64(const std::string& path, const std::vector<std::pair<std::uint64_t,std::uint64_t>>& edges){
  std::ofstream f(path, std::ios::binary); if(!f) die("Cannot open output "+path);
  const char magic[] = "\x93NUMPY"; f.write(magic,6);
  const unsigned char ver[2]={1,0}; f.write((char*)ver,2);
  std::ostringstream dict; dict<<"{'descr': '<i8', 'fortran_order': False, 'shape': ("<<edges.size()<<", 2), }";
  std::string header=dict.str(); size_t base=6+2+2; size_t hlen=header.size()+1; size_t pad=16-((base+hlen)%16); if(pad==16) pad=0;
  header.append(pad,' '); header.push_back('\n'); uint16_t h16=(uint16_t)header.size();
  f.write((char*)&h16,2); f.write(header.data(), header.size());
  for(const auto& e: edges){
    std::int64_t a=(std::int64_t)e.first, b=(std::int64_t)e.second;
    f.write((char*)&a,sizeof(a)); f.write((char*)&b,sizeof(b));
  }
}

// ---------------- Edge extraction helpers ----------------
static inline void dedup_sorted_pairs(std::vector<std::pair<std::uint64_t,std::uint64_t>>& v){
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

// 3D path (optionally parallel with TBB if CGAL_LINKED_WITH_TBB is defined at build time)
struct EdgeComputer3D {
  using K = CGAL::Exact_predicates_inexact_constructions_kernel;
  using Vb0 = CGAL::Triangulation_vertex_base_3<K>;
  using Vb  = CGAL::Triangulation_vertex_base_with_info_3<std::uint64_t, K, Vb0>;
  using Cb  = CGAL::Delaunay_triangulation_cell_base_3<K>;
#ifdef CGAL_LINKED_WITH_TBB
  using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Parallel_tag>;
#else
  using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
#endif
  using DT3 = CGAL::Delaunay_triangulation_3<K, Tds>;
  using P3 = K::Point_3;

  static void run(const std::vector<double>& coords, size_t N, std::vector<std::pair<std::uint64_t,std::uint64_t>>& edges){
    std::vector<std::pair<P3, std::uint64_t>> pts; pts.reserve(N);
    double xmin=+1e300, ymin=+1e300, zmin=+1e300, xmax=-1e300, ymax=-1e300, zmax=-1e300;
    for(size_t i=0;i<N;++i){
      double x=coords[3*i+0], y=coords[3*i+1], z=coords[3*i+2];
      xmin=std::min(xmin,x); ymin=std::min(ymin,y); zmin=std::min(zmin,z);
      xmax=std::max(xmax,x); ymax=std::max(ymax,y); zmax=std::max(zmax,z);
      pts.emplace_back(P3(x,y,z), (std::uint64_t)i);
    }
    CGAL::First_of_pair_property_map<std::pair<P3, std::uint64_t>> first_of_pair;
    using Traits3 = CGAL::Spatial_sort_traits_adapter_3<K, decltype(first_of_pair)>;
    CGAL::spatial_sort(pts.begin(), pts.end(), Traits3(first_of_pair));
#ifdef CGAL_LINKED_WITH_TBB
    std::unique_ptr<typename DT3::Lock_data_structure> lds_holder;
    typename DT3::Lock_data_structure* lds_ptr=nullptr;
    CGAL::Bbox_3 bbox(xmin,ymin,zmin,xmax,ymax,zmax);
    lds_holder.reset(new typename DT3::Lock_data_structure(bbox, 64));
    lds_ptr = lds_holder.get();
    DT3 dt(pts.begin(), pts.end(), lds_ptr);
#else
    DT3 dt(pts.begin(), pts.end());
#endif
    edges.reserve((size_t)dt.number_of_finite_edges());
    for(auto e=dt.finite_edges_begin(); e!=dt.finite_edges_end(); ++e){
      auto c=e->first; int i=e->second, j=e->third;
      auto vi=c->vertex(i), vj=c->vertex(j);
      if(dt.is_infinite(vi) || dt.is_infinite(vj)) continue;
      auto a=vi->info(), b=vj->info(); if(a>b) std::swap(a,b);
      edges.emplace_back(a,b);
    }
    // already unique per iterator, but keep it safe (degeneracies)
    dedup_sorted_pairs(edges);
  }
};

// ND path using dD Delaunay
struct EdgeComputerND {
  using GT  = CGAL::Epick_d<CGAL::Dynamic_dimension_tag>;
  using Vb  = CGAL::Triangulation_vertex<GT, std::uint64_t>;
  using Cb  = CGAL::Triangulation_full_cell<GT>;
  using TDS = CGAL::Triangulation_data_structure<CGAL::Dynamic_dimension_tag, Vb, Cb>;
  using DT  = CGAL::Delaunay_triangulation<GT, TDS>;
  using Point = GT::Point_d;

  static void run(const std::vector<double>& coords, size_t N, int dim, std::vector<std::pair<std::uint64_t,std::uint64_t>>& edges){
    DT dt(dim, GT());
    std::vector<std::pair<Point, std::uint64_t>> pts;
    pts.reserve(N);
    for(size_t i=0;i<N;++i){
      const double* p = coords.data() + i*(size_t)dim;
      pts.emplace_back(Point(p, p+dim), (std::uint64_t)i);
    }
    CGAL::First_of_pair_property_map<std::pair<Point, std::uint64_t>> first_of_pair;
    using Traitsd = CGAL::Spatial_sort_traits_adapter_d<GT, decltype(first_of_pair)>;
    CGAL::spatial_sort(pts.begin(), pts.end(), Traitsd(first_of_pair));
    for(const auto& entry : pts){
      auto vh = dt.insert(entry.first);
      if(vh!=DT::Vertex_handle()) vh->data() = entry.second;
    }
    int cur = dt.current_dimension();
    if(cur<1){ edges.clear(); return; }
    const int verts_per_cell = cur+1;
    const size_t pairs_per_cell = (size_t)verts_per_cell*(verts_per_cell-1)/2;

    std::vector<typename DT::Full_cell_const_handle> cells;
    cells.reserve(dt.number_of_finite_full_cells());
    // `finite_full_cells_begin` returns a filter iterator.  Its base iterator is
    // the compact-container iterator (the actual handle) that we need to store.
    // Use `.base()` to retrieve it so the vector holds valid cell handles.
    for(auto it=dt.finite_full_cells_begin(); it!=dt.finite_full_cells_end(); ++it)
      cells.push_back(it.base());

#ifdef _OPENMP
    int T = omp_get_max_threads();
    std::vector<std::vector<std::pair<std::uint64_t,std::uint64_t>>> tls(T);
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      auto& local = tls[tid];
      local.reserve((cells.size()/T+1)*pairs_per_cell);
      #pragma omp for schedule(static)
      for(long long ci=0; ci<(long long)cells.size(); ++ci){
        auto c = cells[ci];
        std::vector<std::uint64_t> idx(verts_per_cell);
        for(int k=0;k<verts_per_cell;++k) idx[k]=c->vertex(k)->data();
        for(int i=0;i<verts_per_cell;++i) for(int j=i+1;j<verts_per_cell;++j){
          auto a=idx[i], b=idx[j]; if(a>b) std::swap(a,b); local.emplace_back(a,b);
        }
      }
    }
    size_t total=0; for(auto& v:tls) total += v.size();
    edges.reserve(total);
    for(auto& v:tls){ edges.insert(edges.end(), v.begin(), v.end()); v.clear(); v.shrink_to_fit(); }
#else
    edges.reserve(cells.size()*pairs_per_cell);
    for(size_t ci=0; ci<cells.size(); ++ci){
      auto c = cells[ci];
      std::vector<std::uint64_t> idx(verts_per_cell);
      for(int k=0;k<verts_per_cell;++k) idx[k]=c->vertex(k)->data();
      for(int i=0;i<verts_per_cell;++i) for(int j=i+1;j<verts_per_cell;++j){
        auto a=idx[i], b=idx[j]; if(a>b) std::swap(a,b); edges.emplace_back(a,b);
      }
    }
#endif
    dedup_sorted_pairs(edges);
  }
};

int main(int argc, char** argv){
  if(argc!=3){
    std::cerr<<"Usage: "<<argv[0]<<" input_points.npy output_edges.npy\n";
    return 64;
  }
  std::string in_path=argv[1], out_path=argv[2];
  std::vector<double> coords; size_t N=0; int d=0;
  read_npy_matrix<double>(in_path, coords, N, d);
  if(N==0 || d<1){ std::vector<std::pair<std::uint64_t,std::uint64_t>> empty; write_npy_edges_i64(out_path, empty); return 0; }

  std::vector<std::pair<std::uint64_t,std::uint64_t>> edges;
  if(d==3){
    // Try 3D path
    try{
      EdgeComputer3D::run(coords, N, edges);
    }catch(...){
      // Fallback to ND if something odd happens (shouldn't)
      edges.clear();
      EdgeComputerND::run(coords, N, d, edges);
    }
  }else{
    EdgeComputerND::run(coords, N, d, edges);
  }

  write_npy_edges_i64(out_path, edges);
  std::cerr<<"[info] Wrote "<<edges.size()<<" edges\n";
  return 0;
}
