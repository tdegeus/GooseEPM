/**
 * @file version.h
 * @copyright Copyright 2022. Tom de Geus. All rights reserved.
 * @license This project is released under the MIT License.
 */

#ifndef GOOSEEPM_SYSTEM_H
#define GOOSEEPM_SYSTEM_H

#include <prrng.h>
#include <vector>

#include "config.h"
#include "version.h"

namespace GooseEPM {

namespace detail {

/**
 * @brief Create a distance lookup as follows:
 *
 *      [0, 1, 2, ..., N - 2, N - 1, -N + 1, -N + 2, ... -3, -2, -1]
 *
 * with `N` the shape of the propagator along the considered axis (`== distance.size()`).
 * For example for `N = 7` the input could be:
 *
 *      distance = [-3, -2, -1,  0,  1,  2,  3]
 *
 * The the lookup table (output) is:
 *
 *      // dist:    0   1   2   3   4   5   6  -6  -5  -4  -3  -2  -1
 *      lookup = [  3,  4,  5,  6,  0,  1,  2,  4,  5,  6,  0,  1,  2]
 *
 * whereby periodicity is applied such that:
 *
 *      // dist:    0   1   2   3   4   5   6  -6  -5  -4  -3  -2  -1
 *      // per:     0   1   2   3  -3  -2  -1   1   2   3  -3  -2  -1
 *
 * @param distance List with distances according to which an axis of the propagator is orders.
 * @return Look-up list.
 */
template <class T>
inline T create_distance_lookup(const T& distance)
{
    using value_type = typename T::value_type;
    static_assert(std::numeric_limits<value_type>::is_integer, "Distances must be integer");
    static_assert(std::numeric_limits<value_type>::is_signed, "Distances must be signed");

    value_type lower = xt::amin(distance)();
    value_type upper = xt::amax(distance)() + 1;
    auto d = xt::arange<value_type>(lower, upper);
    GOOSEEPM_REQUIRE(xt::all(xt::in1d(distance, d)), std::invalid_argument);
    GOOSEEPM_REQUIRE(distance.size() == upper - lower, std::invalid_argument);

    value_type N = static_cast<value_type>(distance.size());
    T ret = xt::empty<value_type>({2 * N - 1});

    for (value_type i = 0; i < upper; ++i) {
        ret(i) = xt::argmax(xt::equal(distance, i))();
    }
    for (value_type i = upper; i < N; ++i) {
        ret(i) = xt::argmax(xt::equal(distance, i - N))();
    }
    for (value_type i = -1; i >= lower; --i) {
        ret.periodic(i) = xt::argmax(xt::equal(distance, i))();
    }
    for (value_type i = lower; i > -N; --i) {
        ret.periodic(i) = xt::argmax(xt::equal(distance, N + i))();
    }

    return ret;
}

} // namespace detail

class SystemAthermal {

public:
    template <class T, class S, class Y, class Z>
    SystemAthermal(
        const T& propagator,
        const Y& sigmay_mean,
        const Z& sigmay_std,
        const S& sigmay_initstate,
        uint64_t seed,
        double failure_rate,
        double alpha,
        bool fixed_stress)
    {
        GOOSEEPM_REQUIRE(propagator.dimension() == 2, std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_initstate, propagator.shape()), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_mean, propagator.shape()), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_std, propagator.shape()), std::out_of_range);

        m_t = 0;
        m_failure_rate = failure_rate;
        m_alpha = alpha;
        m_fixed_stress = fixed_stress;
        m_propagator = propagator;
        m_sigy_gen = prrng::pcg32_tensor<2>(sigmay_initstate);
        m_gen = prrng::pcg32(seed);
        m_sig = xt::zeros<double>(sigmay_initstate.shape());
        m_epsp = xt::zeros<double>(sigmay_initstate.shape());
        m_sigy = xt::empty<double>(sigmay_initstate.shape());
        m_sigy_mu = sigmay_mean;
        m_sigy_std = sigmay_std;
        m_sigbar = xt::mean(m_sig)();

        for (size_t i = 0; i < m_sigy.size(); ++i) {
            m_sigy.flat(i) = m_sigy_gen.flat(i).normal(
                std::array<size_t, 0>{}, m_sigy_mu.flat(i), m_sigy_std.flat(i))();
        }

        if (m_propagator.shape(0) % 2 == 0) {
            m_imid = m_propagator.shape(0) / 2;
        }
        else {
            m_imid = (m_propagator.shape(0) - 1) / 2;
        }

        if (m_propagator.shape(1) % 2 == 0) {
            m_jmid = m_propagator.shape(1) / 2;
        }
        else {
            m_jmid = (m_propagator.shape(1) - 1) / 2;
        }
    }

    /**
     * @brief Randomise the stress field changing the stress at `(i, j)` by a random value,
     * while changing it by the same value (with a prefactor) at `(i +- delta_r, j +- delta_r)`.
     *
     * @param sigma_std Width of the normal distribution of stresses (mean == 0).
     * @param delta_r Distance to use, see above.
     */
    void initSigmaTrick(double sigma_std, size_t delta_r)
    {
        ptrdiff_t d = static_cast<ptrdiff_t>(delta_r);

        for (ptrdiff_t i = 0; i < m_sig.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_sig.shape(1); ++j) {
                double dsig = m_gen.normal(std::array<size_t, 0>{}, 0, sigma_std)();
                m_sig(i, j) += dsig;
                m_sig.periodic(i - d, j) -= 0.5 * dsig;
                m_sig.periodic(i + d, j) -= 0.5 * dsig;
                m_sig.periodic(i, j - d) -= 0.5 * dsig;
                m_sig.periodic(i, j + d) -= 0.5 * dsig;
                m_sig.periodic(i + d, j + d) += 0.25 * dsig;
                m_sig.periodic(i - d, j + d) += 0.25 * dsig;
                m_sig.periodic(i + d, j - d) += 0.25 * dsig;
                m_sig.periodic(i - d, j - d) += 0.25 * dsig;
            }
        }

        m_sig *= 0.5;
    }

    /**
     * @brief Randomise the stress field changing the stress at `(i, j)` by a random value,
     * and changing is accordingly in the entire surrounding using the propagator.
     *
     * @param sigma_std Width of the normal distribution of stresses (mean == 0).
     */
    void initSigmaPropogator(double sigma_std)
    {
        m_sig.fill(0);

        for (ptrdiff_t i = 0; i < m_sig.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_sig.shape(1); ++j) {

                double dsig = m_gen.normal(std::array<size_t, 0>{}, 0, sigma_std)();

                for (ptrdiff_t k = 0; k < m_sig.shape(0); ++k) {
                    for (ptrdiff_t l = 0; l < m_sig.shape(1); ++l) {
                        if (i == k && j == l) {
                            m_sig(i, j) += dsig;
                        }
                        m_sig(i, j) += m_propagator.periodic(i - k, j - l) * dsig;
                    }
                }
            }
        }
    }

    /**
     * @brief Restore random number generators.
     * @param sigmay_state State of the random number generator for the sigmay.
     * @param state State of the general purpose random number generator.
     */
    template <class T>
    void restore(const T& sigmay_state, uint64_t state)
    {
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_state, m_sigy_gen.shape()), std::out_of_range);
        m_sigy_gen.restore(sigmay_state);
        m_gen.restore(state);
    }

    size_t makeAthermalFailureStep()
    {
        auto failing = xt::argwhere(m_sig < -m_sigy || m_sig > m_sigy);
        size_t nfailing = failing.size();
        m_t += m_gen.exponential(std::array<size_t, 0>{}, m_failure_rate * nfailing)();
        size_t i = m_gen.randint(std::array<size_t, 0>{}, static_cast<size_t>(nfailing - 1))();
        size_t idx = m_sig.shape(0) * failing[i][0] + failing[i][1];
        this->spatialParticleFailure(idx);
        return idx;
    }

    size_t makeWeakestFailureStep()
    {
        size_t idx = xt::argmin(m_sigy - m_sig)();
        double x = m_sigy.flat(idx) - m_sig.flat(idx);

        if (x < 0) {
            m_t += 1.0;
        }
        else {
            m_t += std::exp(std::pow(200.0 * x, m_alpha));
        }

        this->spatialParticleFailure(idx);
        return idx;
    }

    /**
     * @brief Fail a block.
     *
     * -    Change the stress in the block and stabilise it.
     * -    Apply the propagator to change the stress in all 'surrounding' blocks.
     *
     * @param idx Flat index of the block to fail.
     */
    void spatialParticleFailure(size_t idx)
    {
        double dsig = m_sig.flat(idx) + m_gen.normal(std::array<size_t, 0>{}, 0.0, 0.01)();

        m_sig.flat(idx) -= dsig;
        m_epsp.flat(idx) += dsig;
        m_sigy.flat(idx) = m_sigy_gen.flat(idx).normal(
            std::array<size_t, 0>{}, m_sigy_mu.flat(idx), m_sigy_std.flat(idx))();

        auto index = xt::unravel_index(idx, m_sig.shape());
        size_t i0 = index[0];
        size_t j0 = index[1];

        for (size_t i = 0; i < m_sig.shape(0); ++i) {
            for (size_t j = 0; j < m_sig.shape(1); ++j) {
                if (i == i0 && j == j0) {
                    continue;
                }
                size_t di = this->propagator_index(i0, i, m_imid);
                size_t dj = this->propagator_index(j0, j, m_jmid);
                m_sig(i, j) += m_propagator(di, dj) * dsig;
            }
        }

        if (!m_fixed_stress) {
            m_sigbar -= dsig / static_cast<double>(m_sig.size());
        }

        m_sig -= xt::mean(m_sig)() - m_sigbar;
    }

private:
    size_t propagator_index(size_t i0, size_t i, size_t imid)
    {
        if (i < i0) {
            size_t d = i0 - i;
            if (d > imid) {
                return imid + (imid - d);
            }
            else {
                return imid - d;
            }
        }
        else {
            size_t d = i - i0;
            if (d > imid) {
                return imid + (imid - d);
            }
            else {
                return imid + d;
            }
        }
    }

protected:
    prrng::pcg32 m_gen;
    prrng::pcg32_tensor<2> m_sigy_gen;
    array_type::tensor<double, 2> m_propagator;
    array_type::tensor<double, 2> m_sig;
    array_type::tensor<double, 2> m_sigy;
    array_type::tensor<double, 2> m_sigy_mu;
    array_type::tensor<double, 2> m_sigy_std;
    array_type::tensor<double, 2> m_epsp;
    double m_t;
    double m_failure_rate;
    double m_alpha;
    bool m_fixed_stress;
    double m_sigbar;
    size_t m_imid;
    size_t m_jmid;
};

} // namespace GooseEPM

#endif
