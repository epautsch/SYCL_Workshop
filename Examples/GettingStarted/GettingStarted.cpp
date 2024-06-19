#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;


int main() {
	const size_t N = 1024;
	std::vector<int> a(N, 1), b(N, 2), c(N, 0);

	queue q(gpu_selector_v);

	std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";

	buffer<int> bufferA(a.data(), range<1>(N));
	buffer<int> bufferB(b.data(), range<1>(N));
	buffer<int> bufferC(c.data(), range<1>(N));

	q.submit([&](handler& h) {
		accessor accA(bufferA, h, read_only);
		accessor accB(bufferB, h, read_only);
		accessor accC(bufferC, h, write_only, no_init);

		h.parallel_for(range<1>(N), [=](id<1> i) {
			accC[i] = accA[i] * accB[i];
		});
	});

	q.wait();

	for (size_t i = 0; i < N; ++i) {
		if (c[i] != a[i] * b[i]) {
			std::cerr << "Error at index " << i << ": " << c[i] << "\n";
			return 1;
		}
	}

	std::cout << "Vector multiplication successful.\n";
	return 0;
}
