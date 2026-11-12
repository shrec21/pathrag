// Python binding for the Backstitch graph index.
//
// This is the only seam between pathrag and the research code: it wraps
// bgr::Index (build, search, insert) and bgr::Engine (remove, background
// repair) and nothing else. Research changes to repair, triggers or routing
// show up here with no edits, as long as these public signatures hold.
//
// Backstitch aborts on capacity or label overflow, so every such case is
// checked here first and raised as a Python exception instead.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "bgr/index.h"
#include "bgr/pipeline/engine.h"
#include "bgr/repair/repairer.h"

namespace py = pybind11;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using U32 = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

namespace {

class PyIndex {
 public:
  PyIndex(uint32_t dim, uint32_t capacity, uint32_t label_capacity, uint32_t R, uint32_t L_build, float alpha,
          const std::string& version, const std::string& trigger, const std::string& policy, uint32_t workers,
          uint32_t L_d, uint32_t tau_risk)
      : label_capacity_(label_capacity) {
    if (label_capacity == 0) throw std::invalid_argument("label_capacity must be > 0");
    bgr::IndexParams p;
    p.dim = dim;
    p.capacity = capacity + 1;  // slot 0 is the frozen start node
    p.R = R;
    p.R_relaxed = R + 1;  // Greator R' = R + 1
    p.L_build = L_build;
    p.alpha = alpha;
    idx_ = std::make_unique<bgr::Index>(p);
    idx_->reserve_labels(label_capacity);

    cfg_.version = bgr::parse_version(version);
    cfg_.trigger = bgr::parse_trigger(trigger);
    cfg_.workers = workers;
    cfg_.repair.policy = bgr::parse_policy(policy);
    if (L_d) cfg_.repair.L_d = L_d;
    cfg_.repair.tau_risk = tau_risk;
    if (cfg_.version == bgr::Version::Rebuild)
      throw std::invalid_argument("the rebuild version is a benchmark reference, not exposed here");
  }

  ~PyIndex() {
    if (engine_) engine_->stop(false);
  }

  void build(F32 data, U32 labels, uint32_t threads, uint64_t seed) {
    if (built_) throw std::logic_error("build() was already called");
    check_matrix(data, "data");
    const auto n = static_cast<uint32_t>(data.shape(0));
    if (labels.ndim() != 1 || static_cast<uint32_t>(labels.shape(0)) != n)
      throw std::invalid_argument("labels must be a 1-d array with one label per row");
    if (n == 0) throw std::invalid_argument("build needs at least one vector");
    if (n + 1 > idx_->capacity()) throw std::invalid_argument("capacity too small for build");
    auto lab = labels.unchecked<1>();
    std::vector<uint8_t> seen(label_capacity_, 0);
    for (uint32_t i = 0; i < n; ++i) {
      if (lab(i) >= label_capacity_) throw std::invalid_argument("label >= label_capacity");
      if (seen[lab(i)]++) throw std::invalid_argument("duplicate label in build");
    }
    {
      py::gil_scoped_release nogil;
      idx_->build(data.data(), labels.data(), n, threads, seed);
    }
    engine_ = std::make_unique<bgr::Engine>(*idx_, cfg_);
    built_ = true;
  }

  // Top-k live labels per query row; -1 pads rows with fewer than k hits.
  py::array_t<int64_t> search(F32 queries, uint32_t k, uint32_t L) {
    require_built();
    check_matrix(queries, "queries");
    const auto m = static_cast<size_t>(queries.shape(0));
    py::array_t<int64_t> out({m, static_cast<size_t>(k)});
    auto o = out.mutable_unchecked<2>();
    for (size_t i = 0; i < m; ++i)
      for (uint32_t j = 0; j < k; ++j) o(i, j) = -1;
    {
      py::gil_scoped_release nogil;
      bgr::SearchScratch s(idx_->capacity());
      std::vector<uint32_t> hits;
      const float* q = queries.data();
      for (size_t i = 0; i < m; ++i) {
        s.next_epoch();
        idx_->search(q + i * idx_->dim(), k, L, s, hits);
        for (size_t j = 0; j < hits.size(); ++j) o(i, j) = hits[j];
      }
    }
    return out;
  }

  void insert(uint32_t label, F32 x) {
    require_built();
    if (x.ndim() != 1 || static_cast<uint32_t>(x.shape(0)) != idx_->dim())
      throw std::invalid_argument("x must be a 1-d vector of length dim");
    std::lock_guard<std::mutex> g(update_mu_);
    if (label >= label_capacity_) throw std::invalid_argument("label >= label_capacity");
    if (idx_->size() >= idx_->capacity())
      throw std::runtime_error("index is full: deleted slots are not reused, raise capacity");
    const bgr::NodeId cur = idx_->slot_of_label(label);
    if (cur != bgr::kInvalidId && idx_->is_live(cur)) throw std::invalid_argument("label is already live");
    py::gil_scoped_release nogil;
    if (!scratch_) {
      scratch_ = std::make_unique<bgr::SearchScratch>(idx_->capacity());
      prune_ = std::make_unique<bgr::PruneScratch>();
    }
    engine_->insert(label, x.data(), *scratch_, *prune_);
  }

  // Tombstones the label and hands repair to the engine. False if not live.
  bool remove(uint32_t label) {
    require_built();
    std::lock_guard<std::mutex> g(update_mu_);
    py::gil_scoped_release nogil;
    return engine_->remove(label, round_);
  }

  // Round numbers tag repair jobs; the trace runner advances them.
  void next_round() { ++round_; }

  void flush() {
    require_built();
    py::gil_scoped_release nogil;
    engine_->flush_parked(round_);
    engine_->wait_idle();
  }

  bool is_live(uint32_t label) const {
    if (!built_ || label >= label_capacity_) return false;
    const bgr::NodeId s = idx_->slot_of_label(label);
    return s != bgr::kInvalidId && idx_->is_live(s);
  }

  py::dict stats() const {
    py::dict d;
    d["slots_used"] = idx_->size() == 0 ? 0 : idx_->size() - 1;
    d["capacity"] = idx_->capacity() - 1;
    if (engine_) {
      d["deletes"] = engine_->deletes();
      d["repairs_done"] = engine_->repairs_done();
      d["backlog"] = engine_->backlog();
      d["queued"] = engine_->queued();
      d["parked"] = engine_->parked();
      d["unreachable_live"] = idx_->count_unreachable_live();
    }
    d["version"] = bgr::to_string(cfg_.version);
    d["trigger"] = bgr::to_string(cfg_.trigger);
    d["policy"] = bgr::to_string(cfg_.repair.policy);
    return d;
  }

  uint32_t dim() const { return idx_->dim(); }

 private:
  void require_built() const {
    if (!built_) throw std::logic_error("call build() first");
  }
  void check_matrix(const F32& a, const char* name) const {
    if (a.ndim() != 2 || static_cast<uint32_t>(a.shape(1)) != idx_->dim())
      throw std::invalid_argument(std::string(name) + " must be a 2-d array with dim columns");
  }

  std::unique_ptr<bgr::Index> idx_;
  std::unique_ptr<bgr::Engine> engine_;
  bgr::EngineConfig cfg_;
  uint32_t label_capacity_;
  uint32_t round_ = 0;
  bool built_ = false;
  std::mutex update_mu_;  // one update thread, as in the Backstitch runner
  std::unique_ptr<bgr::SearchScratch> scratch_;
  std::unique_ptr<bgr::PruneScratch> prune_;
};

}  // namespace

PYBIND11_MODULE(_backstitch, m) {
  m.doc() = "Backstitch graph vector index with background repair";
  py::class_<PyIndex>(m, "Index")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, const std::string&, const std::string&,
                    const std::string&, uint32_t, uint32_t, uint32_t>(),
           py::arg("dim"), py::arg("capacity"), py::arg("label_capacity"), py::arg("R") = 64,
           py::arg("L_build") = 100, py::arg("alpha") = 1.2f, py::arg("version") = "background",
           py::arg("trigger") = "every_delete", py::arg("policy") = "routed", py::arg("workers") = 2,
           py::arg("L_d") = 0, py::arg("tau_risk") = 0)
      .def("build", &PyIndex::build, py::arg("data"), py::arg("labels"), py::arg("threads") = 1,
           py::arg("seed") = 42)
      .def("search", &PyIndex::search, py::arg("queries"), py::arg("k"), py::arg("L") = 100)
      .def("insert", &PyIndex::insert, py::arg("label"), py::arg("x"))
      .def("remove", &PyIndex::remove, py::arg("label"))
      .def("next_round", &PyIndex::next_round)
      .def("flush", &PyIndex::flush)
      .def("is_live", &PyIndex::is_live, py::arg("label"))
      .def("stats", &PyIndex::stats)
      .def_property_readonly("dim", &PyIndex::dim);
}
