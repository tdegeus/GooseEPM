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
    template <class T, class D, class Y, class Z>
    SystemAthermal(
        const T& propagator,
        const D& dx,
        const D& dy,
        const Y& sigmay_mean,
        const Z& sigmay_std,
        uint64_t seed,
        double failure_rate,
        double alpha,
        double sigmabar,
        bool fixed_stress)
    {
        GOOSEEPM_REQUIRE(propagator.dimension() == 2, std::out_of_range);
        GOOSEEPM_REQUIRE(dx.size() == propagator.shape(0), std::out_of_range);
        GOOSEEPM_REQUIRE(dy.size() == propagator.shape(1), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_mean, propagator.shape()), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_std, propagator.shape()), std::out_of_range);

        m_t = 0;
        m_failure_rate = failure_rate;
        m_alpha = alpha;
        m_fixed_stress = fixed_stress;
        m_propagator = propagator;
        m_dx = detail::create_distance_lookup(dx);
        m_dy = detail::create_distance_lookup(dy);
        m_gen = prrng::pcg32(seed);
        m_sig = sigmabar * xt::ones<double>(propagator.shape());
        m_epsp = xt::zeros<double>(propagator.shape());
        m_sigy = xt::empty<double>(propagator.shape());
        m_sigy_mu = sigmay_mean;
        m_sigy_std = sigmay_std;
        m_sigbar = 0;

        for (size_t i = 0; i < m_sigy.size(); ++i) {
            m_sigy.flat(i) =
                m_gen.normal(std::array<size_t, 0>{}, m_sigy_mu.flat(i), m_sigy_std.flat(i))();
        }
    }

    /**
     * @brief Set the state of the random number generator.
     * @param state State.
     */
    void set_state(uint64_t state)
    {
        m_gen.restore(state);
    }

    /**
     * @brief Get the state of the random number generator.
     * @return State.
     */
    uint64_t state() const
    {
        return m_gen.state();
    }

    /**
     * @brief Set the plastic strain.
     * @param epsp Plastic strain.
     */
    void set_epsp(const array_type::tensor<double, 2>& epsp)
    {
        m_epsp = epsp;
    }

    /**
     * @brief Get the plastic strain.
     * @return Plastic strain.
     */
    const array_type::tensor<double, 2>& epsp() const
    {
        return m_epsp;
    }

    /**
     * @brief Set the yield stress.
     * @param sigmay Yield stress.
     */
    void set_sigmay(const array_type::tensor<double, 2>& sigmay)
    {
        m_sigy = sigmay;
    }

    /**
     * @brief Get the yield stress.
     * @return Yield stress.
     */
    const array_type::tensor<double, 2>& sigmay() const
    {
        return m_sigy;
    }

    /**
     * @brief Set the stress.
     * @param sigma Stress.
     */
    void set_sigma(const array_type::tensor<double, 2>& sigma)
    {
        m_sig = sigma;
    }

    /**
     * @brief Get the stress.
     * @return Stress.
     */
    const array_type::tensor<double, 2>& sigma() const
    {
        return m_sig;
    }

    /**
     * @brief Set the imposed stress.
     * @param sigmabar Imposed stress.
     */
    void set_sigmabar(double sigmabar)
    {
        m_sigbar = sigmabar;
        m_sig -= xt::mean(m_sig)() - m_sigbar;
    }

    /**
     * @brief Get the average (imposed) stress.
     * @return Average stress.
     */
    double sigmabar() const
    {
        return m_sigbar;
    }

    /**
     * @brief Randomise the stress field changing the stress at `(i, j)` by a random value,
     * while changing it by the same value (with a prefactor) at `(i +- delta_r, j +- delta_r)`.
     *
     * @param sigma_std Width of the normal distribution of stresses (mean == 0).
     * @param delta_r Distance to use, see above.
     */
    void initSigmaFast(double sigma_std, size_t delta_r)
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

        m_sig *= 0.5; // todo: clearify why this is needed
    }

    /**
     * @brief Randomise the stress field changing the stress at `(i, j)` by a random value,
     * and changing the entire surrounding based on that value using the propagator.
     *
     * @param sigma_std Width of the normal distribution of stresses (mean == 0).
     */
    void initSigmaPropogator(double sigma_std)
    {
        m_sig.fill(0); // todo: I would think that this is a bug

        for (ptrdiff_t i = 0; i < m_sig.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_sig.shape(1); ++j) {

                double dsig = m_gen.normal(std::array<size_t, 0>{}, 0, sigma_std)();

                for (ptrdiff_t k = 0; k < m_sig.shape(0); ++k) {
                    for (ptrdiff_t l = 0; l < m_sig.shape(1); ++l) {
                        if (i == k && j == l) {
                            m_sig(k, l) += dsig;
                        }
                        else {
                            m_sig(k, l) +=
                                m_propagator(m_dx.periodic(i - k), m_dy.periodic(j - l)) * dsig;
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Make `n` makeFailureStep() calls.
     * @param n Number of steps to make.
     */
    void makeFailureSteps(size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            makeFailureStep();
        }
    }

    /**
     * @brief Failure step.
     * @return Index of the failing particle (flat index).
     */
    size_t makeFailureStep()
    {
        auto failing = xt::argwhere(m_sig < -m_sigy || m_sig > m_sigy);
        size_t nfailing = failing.size();
        m_t += m_gen.exponential(std::array<size_t, 0>{}, m_failure_rate * nfailing)();
        size_t i = m_gen.randint(std::array<size_t, 0>{}, static_cast<size_t>(nfailing - 1))();
        size_t idx = m_sig.shape(0) * failing[i][0] + failing[i][1];
        this->spatialParticleFailure(idx);
        return idx;
    }

    /**
     * @brief Fail weakest particle.
     * @return Index of the failing particle (flat index).
     */
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
     * -    Change the stress in the block.
     * -    Apply the propagator to change the stress in all 'surrounding' blocks.
     *
     * @param idx Flat index of the block to fail.
     */
    void spatialParticleFailure(size_t idx)
    {
        double dsig = m_sig.flat(idx) + m_gen.normal(std::array<size_t, 0>{}, 0.0, 0.01)();

        m_sig.flat(idx) -= dsig;
        m_epsp.flat(idx) += dsig;
        m_sigy.flat(idx) =
            m_gen.normal(std::array<size_t, 0>{}, m_sigy_mu.flat(idx), m_sigy_std.flat(idx))();

        auto index = xt::unravel_index(idx, m_sig.shape());
        ptrdiff_t i0 = static_cast<ptrdiff_t>(index[0]);
        ptrdiff_t j0 = static_cast<ptrdiff_t>(index[1]);

        for (ptrdiff_t i = 0; i < m_sig.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_sig.shape(1); ++j) {
                if (i == i0 && j == j0) {
                    continue;
                }
                m_sig(i, j) += m_propagator(m_dx.periodic(i - i0), m_dx.periodic(j - j0)) * dsig;
            }
        }

        if (!m_fixed_stress) {
            m_sigbar -= dsig / static_cast<double>(m_sig.size());
        }

        m_sig -= xt::mean(m_sig)() - m_sigbar;
    }

    /**
     * @brief Take imposed shear step according the the event-driving protocol.
     */
    void shiftImposedShear()
    {
        double dsig = xt::amin(m_sigy - m_sig)();
        m_sig += dsig;
        m_sigbar += dsig;
    }

protected:
    prrng::pcg32 m_gen;
    array_type::tensor<double, 2> m_propagator;
    array_type::tensor<ptrdiff_t, 1> m_dx;
    array_type::tensor<ptrdiff_t, 1> m_dy;
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
