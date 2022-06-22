#include "py4dgeo/epoch.hpp"
#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include <memory>

namespace py4dgeo {

Epoch::Epoch(const EigenPointCloudRef& cloud_)
  : owned_cloud(nullptr)
  , cloud(cloud_)
  , kdtree(cloud_)
{
}

Epoch::Epoch(std::shared_ptr<EigenPointCloud> cloud_)
  : owned_cloud(cloud_)
  , cloud(*cloud_)
  , kdtree(*cloud_)
{
}

std::ostream&
Epoch::to_stream(std::ostream& stream) const
{
  // Write the cloud itself
  IndexType rows = cloud.rows();
  stream.write(reinterpret_cast<const char*>(&rows), sizeof(IndexType));
  stream.write(reinterpret_cast<const char*>(&cloud(0, 0)),
               sizeof(double) * rows * 3);

  // Write the leaf parameter
  int leaf_parameter = kdtree.leaf_parameter;
  stream.write(reinterpret_cast<const char*>(&leaf_parameter), sizeof(int));

  // Write the search index iff the index was built
  if (leaf_parameter != 0)
    kdtree.search->saveIndex(stream);

  return stream;
}

std::unique_ptr<Epoch>
Epoch::from_stream(std::istream& stream)
{
  // Read the cloud itself
  IndexType rows;
  stream.read(reinterpret_cast<char*>(&rows), sizeof(IndexType));
  auto cloud = std::make_shared<EigenPointCloud>(rows, 3);
  stream.read(reinterpret_cast<char*>(&(*cloud)(0, 0)),
              sizeof(double) * rows * 3);

  // Create the epoch
  auto epoch = std::make_unique<Epoch>(cloud);

  // Read the leaf parameter
  stream.read(reinterpret_cast<char*>(&(epoch->kdtree.leaf_parameter)),
              sizeof(int));

  // Read the search index iff the index was built
  if (epoch->kdtree.leaf_parameter != 0) {
    epoch->kdtree.search = std::make_shared<KDTree::KDTreeImpl>(
      3,
      epoch->kdtree.adaptor,
      nanoflann::KDTreeSingleIndexAdaptorParams(epoch->kdtree.leaf_parameter));
    epoch->kdtree.search->loadIndex(stream);
  }

  return epoch;
}

} // namespace py4dgeo
