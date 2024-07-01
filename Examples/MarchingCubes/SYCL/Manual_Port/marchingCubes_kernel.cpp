#ifndef _MARCHING_CUBES_KERNEL_SYCL_
#define _MARCHING_CUBES_KERNEL_SYCL_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <cstring>
#include "defines.h"
#include "tables.h"

// textures containing look-up tables
dpct::image_wrapper<uint, 1> triTex;
dpct::image_wrapper<uint, 1> numVertsTex;

// volume data
dpct::image_wrapper<uchar, 1> volumeTex;

extern "C" void allocateTextures(uint **d_edgeTable, uint **d_triTable,
                                 uint **d_numVertsTable) {
  *d_edgeTable = static_cast<uint *>(sycl::malloc_device(256 * sizeof(uint), dpct::get_default_queue()));
  dpct::get_default_queue().memcpy(*d_edgeTable, edgeTable, 256 * sizeof(uint)).wait();

  *d_triTable = static_cast<uint *>(sycl::malloc_device(256 * 16 * sizeof(uint), dpct::get_default_queue()));
  dpct::get_default_queue().memcpy(*d_triTable, triTable, 256 * 16 * sizeof(uint)).wait();

  triTex.init(*d_triTable, 256 * 16 * sizeof(uint));

  *d_numVertsTable = static_cast<uint *>(sycl::malloc_device(256 * sizeof(uint), dpct::get_default_queue()));
  dpct::get_default_queue().memcpy(*d_numVertsTable, numVertsTable, 256 * sizeof(uint)).wait();

  numVertsTex.init(*d_numVertsTable, 256 * sizeof(uint));
}

extern "C" void createVolumeTexture(uchar *d_volume, size_t buffSize) {
  volumeTex.init(d_volume, buffSize);
}

extern "C" void destroyAllTextureObjects() {
  triTex.release();
  numVertsTex.release();
  volumeTex.release();
}

float tangle(float x, float y, float z) {
  x *= 3.0f;
  y *= 3.0f;
  z *= 3.0f;
  return (x * x * x * x - 5.0f * x * x + y * y * y * y - 5.0f * y * y +
          z * z * z * z - 5.0f * z * z + 11.8f) * 0.2f + 0.5f;
}

float fieldFunc(float3 p) {
  return tangle(p.x, p.y, p.z);
}

float4 fieldFunc4(float3 p) {
  float v = tangle(p.x, p.y, p.z);
  const float d = 0.001f;
  float dx = tangle(p.x + d, p.y, p.z) - v;
  float dy = tangle(p.x, p.y + d, p.z) - v;
  float dz = tangle(p.x, p.y, p.z + d) - v;
  return float4{dx, dy, dz, v};
}

float sampleVolume(dpct::image_accessor_ext<float, 1> volumeTex, uchar *data, sycl::uint3 p, sycl::uint3 gridSize) {
  p.x = sycl::min(p.x, gridSize.x - 1);
  p.y = sycl::min(p.y, gridSize.y - 1);
  p.z = sycl::min(p.z, gridSize.z - 1);
  uint i = (p.z * gridSize.x * gridSize.y) + (p.y * gridSize.x) + p.x;
  return volumeTex.read(i);
}

sycl::uint3 calcGridPos(uint i, sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask) {
  sycl::uint3 gridPos;
  gridPos.x = i & gridSizeMask.x;
  gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
  gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
  return gridPos;
}

void classifyVoxel(uint *voxelVerts, uint *voxelOccupied, uchar *volume,
                   sycl::uint3 gridSize, sycl::uint3 gridSizeShift,
                   sycl::uint3 gridSizeMask, uint numVoxels,
                   sycl::float3 voxelSize, float isoValue,
                   dpct::image_accessor_ext<uint, 1> numVertsTex,
                   dpct::image_accessor_ext<float, 1> volumeTex,
                   sycl::nd_item<3> item_ct1) {
  uint blockId = item_ct1.get_group(1) * item_ct1.get_group_range(2) + item_ct1.get_group(2);
  uint i = blockId * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

  sycl::uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

  float field[8];
  field[0] = sampleVolume(volumeTex, volume, gridPos, gridSize);
  field[1] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 0, 0), gridSize);
  field[2] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 1, 0), gridSize);
  field[3] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 1, 0), gridSize);
  field[4] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 0, 1), gridSize);
  field[5] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 0, 1), gridSize);
  field[6] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 1, 1), gridSize);
  field[7] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 1, 1), gridSize);

  uint cubeindex;
  cubeindex = uint(field[0] < isoValue);
  cubeindex += uint(field[1] < isoValue) * 2;
  cubeindex += uint(field[2] < isoValue) * 4;
  cubeindex += uint(field[3] < isoValue) * 8;
  cubeindex += uint(field[4] < isoValue) * 16;
  cubeindex += uint(field[5] < isoValue) * 32;
  cubeindex += uint(field[6] < isoValue) * 64;
  cubeindex += uint(field[7] < isoValue) * 128;

  uint numVerts = numVertsTex.read(cubeindex);

  if (i < numVoxels) {
    voxelVerts[i] = numVerts;
    voxelOccupied[i] = (numVerts > 0);
  }
}

extern "C" void launch_classifyVoxel(sycl::queue &q, sycl::range<3> grid,
                                     sycl::range<3> threads, uint *voxelVerts,
                                     uint *voxelOccupied, uchar *volume,
                                     sycl::uint3 gridSize,
                                     sycl::uint3 gridSizeShift,
                                     sycl::uint3 gridSizeMask, uint numVoxels,
                                     sycl::float3 voxelSize, float isoValue) {
  q.submit([&](sycl::handler &cgh) {
    auto numVertsTex_acc = numVertsTex.get_access(cgh);
    auto volumeTex_acc = volumeTex.get_access(cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(grid * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          classifyVoxel(voxelVerts, voxelOccupied, volume, gridSize,
                        gridSizeShift, gridSizeMask, numVoxels, voxelSize,
                        isoValue, numVertsTex_acc, volumeTex_acc, item_ct1);
        });
  }).wait();
}

void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied,
                   uint *voxelOccupiedScan, uint numVoxels,
                   sycl::nd_item<3> item_ct1) {
  uint blockId = item_ct1.get_group(1) * item_ct1.get_group_range(2) + item_ct1.get_group(2);
  uint i = blockId * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

  if (voxelOccupied[i] && (i < numVoxels)) {
    compactedVoxelArray[voxelOccupiedScan[i]] = i;
  }
}

extern "C" void launch_compactVoxels(sycl::queue &q, sycl::range<3> grid,
                                     sycl::range<3> threads,
                                     uint *compactedVoxelArray,
                                     uint *voxelOccupied,
                                     uint *voxelOccupiedScan, uint numVoxels) {
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          compactVoxels(compactedVoxelArray, voxelOccupied,
                        voxelOccupiedScan, numVoxels, item_ct1);
        });
  }).wait();
}

float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1) {
  float t = (isolevel - f0) / (f1 - f0);
  return p0 + t * (p1 - p0);
}

void generateTriangles(float4 *pos, float4 *norm, uint *compactedVoxelArray,
                       uint *numVertsScanned, sycl::uint3 gridSize,
                       sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask,
                       sycl::float3 voxelSize, float isoValue,
                       uint activeVoxels, uint maxVerts,
                       dpct::image_accessor_ext<uint, 1> triTex,
                       dpct::image_accessor_ext<uint, 1> numVertsTex,
                       sycl::nd_item<3> item_ct1) {
  uint blockId = item_ct1.get_group(1) * item_ct1.get_group_range(2) + item_ct1.get_group(2);
  uint i = blockId * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

  if (i > activeVoxels - 1) {
    i = activeVoxels - 1;
  }

  uint voxel = compactedVoxelArray[i];

  sycl::uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

  sycl::float3 p;
  p.x = -1.0f + (gridPos.x * voxelSize.x);
  p.y = -1.0f + (gridPos.y * voxelSize.y);
  p.z = -1.0f + (gridPos.z * voxelSize.z);

  sycl::float3 v[8];
  v[0] = p;
  v[1] = p + sycl::float3(voxelSize.x, 0, 0);
  v[2] = p + sycl::float3(voxelSize.x, voxelSize.y, 0);
  v[3] = p + sycl::float3(0, voxelSize.y, 0);
  v[4] = p + sycl::float3(0, 0, voxelSize.z);
  v[5] = p + sycl::float3(voxelSize.x, 0, voxelSize.z);
  v[6] = p + sycl::float3(voxelSize.x, voxelSize.y, voxelSize.z);
  v[7] = p + sycl::float3(0, voxelSize.y, voxelSize.z);

  float4 field[8];
  field[0] = fieldFunc4(v[0]);
  field[1] = fieldFunc4(v[1]);
  field[2] = fieldFunc4(v[2]);
  field[3] = fieldFunc4(v[3]);
  field[4] = fieldFunc4(v[4]);
  field[5] = fieldFunc4(v[5]);
  field[6] = fieldFunc4(v[6]);
  field[7] = fieldFunc4(v[7]);

  uint cubeindex;
  cubeindex = uint(field[0].w < isoValue);
  cubeindex += uint(field[1].w < isoValue) * 2;
  cubeindex += uint(field[2].w < isoValue) * 4;
  cubeindex += uint(field[3].w < isoValue) * 8;
  cubeindex += uint(field[4].w < isoValue) * 16;
  cubeindex += uint(field[5].w < isoValue) * 32;
  cubeindex += uint(field[6].w < isoValue) * 64;
  cubeindex += uint(field[7].w < isoValue) * 128;

  float3 vertlist[12];
  float3 normlist[12];

  vertexInterp2(isoValue, v[0], v[1], field[0], field[1], vertlist[0], normlist[0]);
  vertexInterp2(isoValue, v[1], v[2], field[1], field[2], vertlist[1], normlist[1]);
  vertexInterp2(isoValue, v[2], v[3], field[2], field[3], vertlist[2], normlist[2]);
  vertexInterp2(isoValue, v[3], v[0], field[3], field[0], vertlist[3], normlist[3]);
  vertexInterp2(isoValue, v[4], v[5], field[4], field[5], vertlist[4], normlist[4]);
  vertexInterp2(isoValue, v[5], v[6], field[5], field[6], vertlist[5], normlist[5]);
  vertexInterp2(isoValue, v[6], v[7], field[6], field[7], vertlist[6], normlist[6]);
  vertexInterp2(isoValue, v[7], v[4], field[7], field[4], vertlist[7], normlist[7]);
  vertexInterp2(isoValue, v[0], v[4], field[0], field[4], vertlist[8], normlist[8]);
  vertexInterp2(isoValue, v[1], v[5], field[1], field[5], vertlist[9], normlist[9]);
  vertexInterp2(isoValue, v[2], v[6], field[2], field[6], vertlist[10], normlist[10]);
  vertexInterp2(isoValue, v[3], v[7], field[3], field[7], vertlist[11], normlist[11]);

  uint numVerts = numVertsTex.read(cubeindex);

  for (int i = 0; i < numVerts; i++) {
    uint edge = triTex.read(cubeindex * 16 + i);

    uint index = numVertsScanned[voxel] + i;

    if (index < maxVerts) {
      pos[index] = float4{vertlist[edge].x, vertlist[edge].y, vertlist[edge].z, 1.0f};
      norm[index] = float4{normlist[edge].x, normlist[edge].y, normlist[edge].z, 0.0f};
    }
  }
}

extern "C" void launch_generateTriangles(sycl::queue &q, sycl::range<3> grid,
                                         sycl::range<3> threads, float4 *pos,
                                         float4 *norm, uint *compactedVoxelArray,
                                         uint *numVertsScanned, sycl::uint3 gridSize,
                                         sycl::uint3 gridSizeShift,
                                         sycl::uint3 gridSizeMask,
                                         sycl::float3 voxelSize, float isoValue,
                                         uint activeVoxels, uint maxVerts) {
  q.submit([&](sycl::handler &cgh) {
    auto triTex_acc = triTex.get_access(cgh);
    auto numVertsTex_acc = numVertsTex.get_access(cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(grid * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          generateTriangles(pos, norm, compactedVoxelArray, numVertsScanned,
                            gridSize, gridSizeShift, gridSizeMask, voxelSize,
                            isoValue, activeVoxels, maxVerts, triTex_acc,
                            numVertsTex_acc, item_ct1);
        });
  }).wait();
}

extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input,
                                  unsigned int numElements) {
  std::exclusive_scan(oneapi::dpl::execution::dpcpp_default, input, input + numElements, output, 0);
}

#endif

