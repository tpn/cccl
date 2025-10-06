#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <typeinfo>

#include <cuda_runtime.h>

#include <unittest/unittest.h>

// Additional debug notes:
// - Printing of input ranges when divergence detected.
// - Independent expected size computation.
// - Device copy sanity checks when B is empty.
// - Mini reproduction and now expanded alternate invocation diagnostics.
// - Added device properties dump and cudaGetLastError() checks.

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator
set_difference(my_system& system, InputIterator1, InputIterator1, InputIterator2, InputIterator2, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestSetDifferenceDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::set_difference(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSetDifferenceDispatchExplicit);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator
set_difference(my_tag, InputIterator1, InputIterator1, InputIterator2, InputIterator2, OutputIterator result)
{
  *result = 13;
  return result;
}

void TestSetDifferenceDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::set_difference(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSetDifferenceDispatchImplicit);

template <typename Vector>
void TestSetDifferenceSimple()
{
  using Iterator = typename Vector::iterator;

  Vector a{0, 2, 4, 5}, b{0, 3, 3, 4, 6};
  Vector ref{2, 5};
  Vector result(2);

  Iterator end = thrust::set_difference(a.begin(), a.end(), b.begin(), b.end(), result.begin());

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}
DECLARE_VECTOR_UNITTEST(TestSetDifferenceSimple);

template <typename T>
static size_t host_expected_difference_size(const thrust::host_vector<T>& A, const thrust::host_vector<T>& B_subset)
{
  size_t count = 0;
  size_t i = 0, j = 0;
  while (i < A.size() && j < B_subset.size())
  {
    if (A[i] < B_subset[j])
    {
      ++count;
      ++i;
    }
    else if (B_subset[j] < A[i])
    {
      ++j;
    }
    else
    {
      ++i;
      ++j;
    }
  }
  count += (A.size() - i);
  return count;
}

template <typename T>
void TestSetDifference(const size_t n)
{
  size_t sizes[]   = {0, 1, n / 2, n, n + 1, 2 * n};
  size_t num_sizes = sizeof(sizes) / sizeof(size_t);

  thrust::host_vector<T> random =
    unittest::random_integers<unittest::int8_t>(n + *thrust::max_element(sizes, sizes + num_sizes));

  thrust::host_vector<T> h_a(random.begin(), random.begin() + n);
  thrust::host_vector<T> h_b(random.begin() + n, random.end());

  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  const char* type_name = typeid(T).name();

  std::cout << "[DEBUG][TestSetDifference] T=" << type_name << " n=" << n << std::endl;

  static bool printed_device_info = false;
  if (!printed_device_info)
  {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::cout
      << "[DEBUG][Env] Device " << dev << " name='" << prop.name << "' cc=" << prop.major << prop.minor
      << " multiProcessorCount=" << prop.multiProcessorCount << " driver=" << (cudaDriverGetVersion(&dev), dev)
      << std::endl;
    printed_device_info = true;
  }

  for (size_t i = 0; i < num_sizes; i++)
  {
    size_t size = sizes[i];

    thrust::host_vector<T> h_result(n + size);
    thrust::device_vector<T> d_result(n + size);

    size_t initial_h_capacity = h_result.size();
    size_t initial_d_capacity = d_result.size();

    std::cout
      << "[DEBUG][TestSetDifference] T=" << type_name << " Iteration=" << i << " size=" << size
      << " n+size=" << n + size << " h_cap=" << initial_h_capacity << " d_cap=" << initial_d_capacity << std::endl;

    typename thrust::host_vector<T>::iterator h_end;
    typename thrust::device_vector<T>::iterator d_end;

    cudaGetLastError(); // clear prior
    h_end = thrust::set_difference(h_a.begin(), h_a.end(), h_b.begin(), h_b.begin() + size, h_result.begin());
    cudaError_t err_after_host_like = cudaGetLastError();
    size_t computed_h_size          = static_cast<size_t>(h_end - h_result.begin());

    cudaGetLastError();
    d_end = thrust::set_difference(d_a.begin(), d_a.end(), d_b.begin(), d_b.begin() + size, d_result.begin());
    cudaError_t err_after_device = cudaGetLastError();
    size_t computed_d_size       = static_cast<size_t>(d_end - d_result.begin());

    thrust::host_vector<T> h_b_subset(h_b.begin(), h_b.begin() + size);
    size_t expected_size = host_expected_difference_size(h_a, h_b_subset);

    if (computed_h_size != computed_d_size || err_after_device != cudaSuccess)
    {
      std::cout
        << "[DEBUG][TestSetDifference] Pre-resize size divergence/err. T=" << type_name << " Iter=" << i
        << " subset_size=" << size << " h_cap=" << initial_h_capacity << " d_cap=" << initial_d_capacity
        << " h_size=" << computed_h_size << " d_size=" << computed_d_size << " expected=" << expected_size
        << " err_host_call=" << (int) err_after_host_like << " err_dev_call=" << (int) err_after_device << std::endl;
    }
    else
    {
      std::cout
        << "[DEBUG][TestSetDifference] Pre-resize agreement. T=" << type_name << " Iter=" << i
        << " subset_size=" << size << " h_cap=" << initial_h_capacity << " d_cap=" << initial_d_capacity
        << " h_size=" << computed_h_size << " d_size=" << computed_d_size << " expected=" << expected_size << std::endl;
    }

    h_result.resize(computed_h_size);
    d_result.resize(computed_d_size);

    bool divergence = (h_result.size() != d_result.size()) || (h_result != d_result) || err_after_device != cudaSuccess;
    if (divergence)
    {
      auto print_input = [](const char* label, const thrust::host_vector<T>& v) {
        const size_t limit = 32;
        std::cout << "  Input " << label << " (size=" << v.size() << "): ";
        for (size_t k = 0; k < v.size() && k < limit; ++k)
        {
          std::cout << v[k];
          if (k + 1 < v.size() && k + 1 < limit)
          {
            std::cout << ", ";
          }
        }
        if (v.size() > limit)
        {
          std::cout << " ...";
        }
        std::cout << std::endl;
      };
      print_input("A", h_a);
      print_input("B(subset)", h_b_subset);

      if (size == 0)
      {
        thrust::device_vector<T> copy_probe(h_a.size());
        cudaGetLastError();
        auto copy_end                    = thrust::copy(d_a.begin(), d_a.end(), copy_probe.begin());
        cudaError_t err_copy             = cudaGetLastError();
        size_t copy_count                = static_cast<size_t>(copy_end - copy_probe.begin());
        thrust::host_vector<T> copy_host = copy_probe;
        std::cout
          << "  [DEBUG] size==0 copy sanity: count=" << copy_count << " copy_err=" << (int) err_copy
          << (copy_host.empty() ? " first=<empty>"
                                : (std::string(" first=") + std::to_string(static_cast<double>(copy_host[0]))))
          << std::endl;
      }

      // Alternate invocation diagnostics (once per iteration when divergence occurs)
      thrust::device_vector<T> alt_out(n + size);
      cudaGetLastError();
      auto alt_end = thrust::set_difference(
        thrust::device, d_a.begin(), d_a.end(), d_b.begin(), d_b.begin() + size, alt_out.begin());
      cudaError_t err_alt = cudaGetLastError();
      size_t alt_size     = static_cast<size_t>(alt_end - alt_out.begin());

      thrust::device_vector<T> alt_out_cmp(n + size);
      cudaGetLastError();
      auto alt_end_cmp = thrust::set_difference(
        thrust::device,
        d_a.begin(),
        d_a.end(),
        d_b.begin(),
        d_b.begin() + size,
        alt_out_cmp.begin(),
        cuda::std::less<T>());
      cudaError_t err_alt_cmp = cudaGetLastError();
      size_t alt_size_cmp     = static_cast<size_t>(alt_end_cmp - alt_out_cmp.begin());

      // Raw pointer variant
      thrust::device_vector<T> raw_out(n + size);
      cudaGetLastError();
      auto raw_end = thrust::set_difference(
        thrust::device,
        thrust::device_pointer_cast(thrust::raw_pointer_cast(d_a.data())),
        thrust::device_pointer_cast(thrust::raw_pointer_cast(d_a.data()) + d_a.size()),
        thrust::device_pointer_cast(thrust::raw_pointer_cast(d_b.data())),
        thrust::device_pointer_cast(thrust::raw_pointer_cast(d_b.data()) + size),
        thrust::device_pointer_cast(thrust::raw_pointer_cast(raw_out.data())));
      cudaError_t err_raw = cudaGetLastError();
      size_t raw_size =
        static_cast<size_t>(raw_end - thrust::device_pointer_cast(thrust::raw_pointer_cast(raw_out.data())));

      std::cout
        << "  [DEBUG] alt_sizes: default_dev=" << computed_d_size << " alt_policy=" << alt_size
        << " alt_cmp=" << alt_size_cmp << " raw_ptr=" << raw_size << " errs(dev/pol/cmp/raw)=" << (int) err_after_device
        << '/' << (int) err_alt << '/' << (int) err_alt_cmp << '/' << (int) err_raw << std::endl;

      if (i == 0)
      {
        thrust::device_vector<T> mini_a(h_a.begin(), h_a.begin() + std::min<size_t>(h_a.size(), 4));
        thrust::device_vector<T> mini_b;
        thrust::device_vector<T> mini_out(mini_a.size());
        cudaGetLastError();
        auto mini_end =
          thrust::set_difference(mini_a.begin(), mini_a.end(), mini_b.begin(), mini_b.end(), mini_out.begin());
        cudaError_t err_mini                 = cudaGetLastError();
        size_t mini_size                     = static_cast<size_t>(mini_end - mini_out.begin());
        thrust::host_vector<T> mini_out_host = mini_out;
        std::cout << "  [DEBUG] Mini reproduction: size=" << mini_size << " err=" << (int) err_mini << " vals:";
        for (size_t k = 0; k < mini_size; ++k)
        {
          std::cout << ' ' << mini_out_host[k];
        }
        std::cout << std::endl;
      }
    }
    else
    {
      // No divergence; print equivalent levels of helpful debugging so
      // we can see what *does* work.
      std::cout
        << "[DEBUG][TestSetDifference] Post-resize agreement. T=" << type_name << " Iter=" << i << " size=" << size
        << " h_cap=" << initial_h_capacity << " d_cap=" << initial_d_capacity << " h_size=" << h_result.size()
        << " d_size=" << d_result.size() << " expected=" << expected_size << std::endl;
    }

    if (h_result.size() != d_result.size())
    {
      std::cout
        << "[DEBUG][TestSetDifference] Post-resize SIZE MISMATCH. T=" << type_name << " Iteration=" << i
        << " input_size=" << size << " h_size=" << h_result.size() << " d_size=" << d_result.size()
        << " expected=" << expected_size << std::endl;

      auto print_sequence = [](const char* label, const thrust::host_vector<T>& seq) {
        const size_t limit = 64;
        std::cout << "  " << label << " (size=" << seq.size() << "): ";
        if (seq.empty())
        {
          std::cout << "<empty>";
        }
        else
        {
          for (size_t k = 0; k < seq.size() && k < limit; ++k)
          {
            std::cout << seq[k];
            if (k + 1 < seq.size() && k + 1 < limit)
            {
              std::cout << ", ";
            }
          }
          if (seq.size() > limit)
          {
            std::cout << " ...";
          }
        }
        std::cout << std::endl;
      };
      thrust::host_vector<T> d_host = d_result;
      print_sequence("host", h_result);
      print_sequence("device", d_host);
    }

    if (h_result != d_result)
    {
      std::cout << "[DEBUG][TestSetDifference] CONTENT MISMATCH. T=" << type_name << " Iter=" << i << std::endl;
    }
    else
    {
      std::cout << "[DEBUG][TestSetDifference] Content match. T=" << type_name << " Iter=" << i << std::endl;
    }

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_VARIABLE_UNITTEST(TestSetDifference);

template <typename T>
void TestSetDifferenceEquivalentRanges(const size_t n)
{
  thrust::host_vector<T> temp = unittest::random_integers<T>(n);
  thrust::host_vector<T> h_a  = temp;
  thrust::sort(h_a.begin(), h_a.end());
  thrust::host_vector<T> h_b = h_a;

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;

  h_end = thrust::set_difference(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
  h_result.resize(h_end - h_result.begin());

  d_end = thrust::set_difference(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

  d_result.resize(d_end - d_result.begin());

  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestSetDifferenceEquivalentRanges);

template <typename T>
void TestSetDifferenceMultiset(const size_t n)
{
  thrust::host_vector<T> vec = unittest::random_integers<int>(2 * n);

  // restrict elements to [min,13)
  for (typename thrust::host_vector<T>::iterator i = vec.begin(); i != vec.end(); ++i)
  {
    int temp = static_cast<int>(*i);
    temp %= 13;
    *i = temp;
  }

  thrust::host_vector<T> h_a(vec.begin(), vec.begin() + n);
  thrust::host_vector<T> h_b(vec.begin() + n, vec.end());

  thrust::sort(h_a.begin(), h_a.end());
  thrust::sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;

  h_end = thrust::set_difference(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
  h_result.resize(h_end - h_result.begin());

  d_end = thrust::set_difference(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

  d_result.resize(d_end - d_result.begin());

  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestSetDifferenceMultiset);

// FIXME: disabled on Windows, because it causes a failure on the internal CI system in one specific configuration.
// That failure will be tracked in a new NVBug, this is disabled to unblock submitting all the other changes.
#if !_CCCL_COMPILER(MSVC)
void TestSetDifferenceWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(0);
  thrust::counting_iterator<long long> end        = begin + (1ll << magnitude);
  thrust::counting_iterator<long long> end_longer = end + 1;
  ASSERT_EQUAL(::cuda::std::distance(begin, end), 1ll << magnitude);

  thrust::device_vector<long long> result;
  result.resize(1);
  thrust::set_difference(thrust::device, begin, end_longer, begin, end, result.begin());

  thrust::host_vector<long long> expected;
  expected.push_back(*end);

  ASSERT_EQUAL(result, expected);
}

void TestSetDifferenceWithBigIndexes()
{
#  ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestSetDifferenceWithBigIndexesHelper(30);
  TestSetDifferenceWithBigIndexesHelper(31);
  TestSetDifferenceWithBigIndexesHelper(32);
  TestSetDifferenceWithBigIndexesHelper(33);
#  endif
}
DECLARE_UNITTEST(TestSetDifferenceWithBigIndexes);

#endif
