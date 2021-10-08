#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include <iostream>

namespace py4dgeo {

inline std::size_t
KDTree::Adaptor::kdtree_get_point_count() const
{
  return cloud.rows();
}

inline double
KDTree::Adaptor::kdtree_get_pt(const IndexType idx, const IndexType dim) const
{
  return cloud(idx, dim);
}

template<class BBOX>
bool
KDTree::Adaptor::kdtree_get_bbox(BBOX&) const
{
  return false;
}

inline std::size_t
KDTree::NoDistancesReturnSet::size() const
{
  return indices.size();
}

inline bool
KDTree::NoDistancesReturnSet::full() const
{
  return true;
}

inline bool
KDTree::NoDistancesReturnSet::addPoint(double dist, IndexType idx)
{
  if (dist < radius)
    indices.push_back(idx);
  return true;
}

inline double
KDTree::NoDistancesReturnSet::worstDist() const
{
  return radius;
}

KDTree::KDTree(const EigenPointCloudRef& cloud)
  : adaptor{ nullptr, cloud }
{}

KDTree::KDTree(const std::shared_ptr<EigenPointCloud>& data)
  : adaptor{ data, *data }
{}

KDTree
KDTree::create(const EigenPointCloudRef& cloud)
{
  return KDTree(cloud);
}

std::unique_ptr<KDTree>
KDTree::from_stream(std::istream& stream)
{
  // Read the cloud itself
  IndexType rows;
  stream.read(reinterpret_cast<char*>(&rows), sizeof(IndexType));
  auto cloud = std::make_shared<EigenPointCloud>(rows, 3);
  stream.read(reinterpret_cast<char*>(&(*cloud)(0, 0)),
              sizeof(double) * rows * 3);
  std::unique_ptr<KDTree> obj(new KDTree(cloud));

  // Read the leaf parameter
  stream.read(reinterpret_cast<char*>(&(obj->leaf_parameter)), sizeof(int));

  // Read the search index iff the index was built
  if (obj->leaf_parameter != 0) {
    obj->search = std::make_shared<KDTreeImpl>(
      3,
      obj->adaptor,
      nanoflann::KDTreeSingleIndexAdaptorParams(obj->leaf_parameter));
    obj->search->loadIndex(stream);
  }

  // Read the precomputation results
  std::size_t size;
  stream.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
  obj->precomputed_indices.resize(size);
  obj->precomputed_distances.resize(size);
  for (std::size_t i = 0; i < size; ++i) {
    std::size_t size2;
    stream.read(reinterpret_cast<char*>(&size2), sizeof(std::size_t));
    obj->precomputed_indices[i].resize(size2);
    obj->precomputed_distances[i].resize(size2);
    stream.read(reinterpret_cast<char*>(obj->precomputed_indices[i].data()),
                sizeof(IndexType) * size2);
    stream.read(reinterpret_cast<char*>(obj->precomputed_distances[i].data()),
                sizeof(double) * size2);
  }

  return obj;
}

std::ostream&
KDTree::to_stream(std::ostream& stream) const
{
  // Write the cloud itself. This is very unfortunate as it is a redundant
  // copy of the point cloud, but this seems to be the only way to have an
  // unpickled copy be actually usable. Scipy does exactly the same.
  IndexType rows = adaptor.cloud.rows();
  stream.write(reinterpret_cast<const char*>(&rows), sizeof(IndexType));
  stream.write(reinterpret_cast<const char*>(&adaptor.cloud(0, 0)),
               sizeof(double) * rows * 3);

  // Write the leaf parameter
  stream.write(reinterpret_cast<const char*>(&leaf_parameter), sizeof(int));

  // Write the search index iff the index was built
  if (leaf_parameter != 0)
    search->saveIndex(stream);

  // Write the precomputation results
  std::size_t size = precomputed_indices.size();
  stream.write(reinterpret_cast<const char*>(&size), sizeof(std::size_t));
  for (std::size_t i = 0; i < size; ++i) {
    std::size_t size2 = precomputed_indices[i].size();
    stream.write(reinterpret_cast<const char*>(&size2), sizeof(std::size_t));
    stream.write(reinterpret_cast<const char*>(precomputed_indices[i].data()),
                 sizeof(IndexType) * size2);
    stream.write(reinterpret_cast<const char*>(precomputed_distances[i].data()),
                 sizeof(double) * size2);
  }

  return stream;
}

void
KDTree::build_tree(int leaf)
{
  search = std::make_shared<KDTreeImpl>(
    3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(leaf));
  search->buildIndex();
  leaf_parameter = leaf;
}

void
KDTree::precompute(EigenPointCloudRef querypoints, double maxradius)
{
  // Resize the output data structures
  precomputed_indices.resize(querypoints.rows());
  precomputed_distances.resize(querypoints.rows());

  // Loop over query points and evaluate with maxradius
  for (IndexType i = 0; i < querypoints.rows(); ++i) {
    RadiusSearchDistanceResult result;
    radius_search_with_distances(&querypoints(i, 0), maxradius, result);

    precomputed_indices[i].resize(result.size());
    precomputed_distances[i].resize(result.size());

    for (std::size_t j = 0; j < result.size(); ++j) {
      precomputed_indices[i][j] = result[j].first;
      precomputed_distances[i][j] = result[j].second;
    }
  }
}

std::size_t
KDTree::radius_search(const double* query,
                      double radius,
                      RadiusSearchResult& result) const
{
  NoDistancesReturnSet set{ radius * radius, result };
  nanoflann::SearchParams params;
  params.sorted = false;
  return search->radiusSearchCustomCallback(query, set, params);
}

std::size_t
KDTree::radius_search_with_distances(const double* query,
                                     double radius,
                                     RadiusSearchDistanceResult& result) const
{
  nanoflann::SearchParams params;
  return search->radiusSearch(query, radius * radius, result, params);
}

std::size_t
KDTree::precomputed_radius_search(const IndexType idx,
                                  double radius,
                                  RadiusSearchResult& result) const
{
  result.clear();

  auto it = std::find_if(precomputed_distances[idx].begin(),
                         precomputed_distances[idx].end(),
                         [radius](auto d) { return d > radius * radius; });

  std::copy(precomputed_indices[idx].begin(),
            precomputed_indices[idx].begin() +
              (it - precomputed_distances[idx].begin()),
            std::back_inserter(result));
  return result.size();
}

} // namespace py4dgeo